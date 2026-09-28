import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Frozen Signal Scanner", page_icon="📈", layout="centered")
NY = ZoneInfo("America/New_York")

# ---------------- shared helpers ----------------
def nyse_sessions(start, end):
    cal = mcal.get_calendar("NYSE")
    return cal.schedule(start_date=start, end_date=end).index.tz_localize(None).normalize()

def nth_session(entry_day, n):
    ss = nyse_sessions(entry_day, pd.Timestamp(entry_day) + pd.Timedelta(days=45))
    ss = ss[ss >= pd.Timestamp(entry_day).normalize()]
    return ss[n-1] if len(ss) >= n else None

def completed_today_cutoff(d):
    now = datetime.now(NY)
    today = pd.Timestamp(now.date())
    return d[d["date"] < today].copy() if now.time() < time(16, 5) else d.copy()

# ---------------- IBM frozen strategy ----------------
def ibm_structure(raw):
    x = raw.copy().reset_index(drop=True)
    ph=(x.high.shift(2)>x.high.shift(3))&(x.high.shift(2)>x.high.shift(4))&(x.high.shift(2)>=x.high.shift(1))&(x.high.shift(2)>=x.high)
    pl=(x.low.shift(2)<x.low.shift(3))&(x.low.shift(2)<x.low.shift(4))&(x.low.shift(2)<=x.low.shift(1))&(x.low.shift(2)<=x.low)
    x["new_ph"]=ph.fillna(False); x["new_pl"]=pl.fillna(False)
    x["pivot_hi"]=np.where(x.new_ph,x.high.shift(2),np.nan)
    x["pivot_lo"]=np.where(x.new_pl,x.low.shift(2),np.nan)
    lh=phh=ll=pll=np.nan; rows=[]
    for _,r in x.iterrows():
        if r.new_ph: phh,lh=lh,r.pivot_hi
        if r.new_pl: pll,ll=ll,r.pivot_lo
        rows.append((lh,phh,ll,pll))
    x[["last_hi","prev_hi","last_lo","prev_lo"]]=pd.DataFrame(rows,index=x.index)
    x["hi_state"]=pd.Series(np.where(x.new_ph,np.where(x.last_hi>x.prev_hi,"HH","LH"),None)).ffill()
    x["lo_state"]=pd.Series(np.where(x.new_pl,np.where(x.last_lo>x.prev_lo,"HL","LL"),None)).ffill()
    x["bull_struct"]=(x.hi_state=="HH")&(x.lo_state=="HL")
    x["bear_struct"]=(x.hi_state=="LH")&(x.lo_state=="LL")
    return x

def ibm_daily(h):
    x=h.copy(); x["day"]=x.date.dt.normalize()
    return x.groupby("day",as_index=False).agg(date=("day","first"),open=("open","first"),
        high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"))

def ibm_fourhour(h):
    x=h.copy(); x["day"]=x.date.dt.normalize()
    x["slot"]=x.groupby("day").cumcount()//4
    return x.groupby(["day","slot"],as_index=False).agg(date=("date","last"),open=("open","first"),
        high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).drop(columns=["slot"])

@st.cache_data(ttl=900)
def ibm_fetch():
    z=yf.download("IBM",period="729d",interval="1h",auto_adjust=False,prepost=False,progress=False,threads=False)
    if z.empty: raise RuntimeError("Yahoo returned no IBM hourly data.")
    if isinstance(z.columns,pd.MultiIndex): z.columns=z.columns.get_level_values(0)
    z=z.reset_index(); dc="Datetime" if "Datetime" in z.columns else "Date"
    dt=pd.to_datetime(z[dc],utc=True).dt.tz_convert(NY)
    z=pd.DataFrame({"date":dt.dt.tz_localize(None),"open":z["Open"],"high":z["High"],
                    "low":z["Low"],"close":z["Close"],"volume":z["Volume"]})
    z=z.dropna().sort_values("date")
    tm=z.date.dt.time
    return z[(tm>=time(9,30))&(tm<time(16,0))].reset_index(drop=True)

def ibm_build(h):
    d=ibm_structure(ibm_daily(h)); q=ibm_structure(ibm_fourhour(h))
    q["day"]=q.date.dt.normalize(); ql=q.groupby("day").tail(1).set_index("day")
    d["day"]=d.date.dt.normalize()
    d["h4_hi"]=d.day.map(ql.hi_state); d["h4_lo"]=d.day.map(ql.lo_state)
    d["h4_bull"]=d.day.map(ql.bull_struct).fillna(False).astype(bool)
    d["h4_swing_low"]=d.day.map(ql.last_lo)
    d["signal"]=d.bear_struct & d.h4_bull
    return d

def scan_ibm():
    h=ibm_fetch(); d=completed_today_cutoff(ibm_build(h))
    if d.empty: raise RuntimeError("No completed IBM session.")
    r=d.iloc[-1]; day=pd.Timestamp(r.date).normalize()
    sl=float(r.h4_swing_low) if pd.notna(r.h4_swing_low) else np.nan
    status="NO SIGNAL"; detail="No frozen IBM pattern."
    if bool(r.signal) and np.isfinite(sl):
        future=nyse_sessions(day+pd.Timedelta(days=1),day+pd.Timedelta(days=45))
        ed=future[0] if len(future) else None
        status="SIGNAL"; detail=f"LONG next open | SL ${sl:.2f} | TP = entry + 2R"
        if ed is not None:
            detail += f" | expected entry {ed.date()} | max exit {nth_session(ed,10).date()}"
    elif len(d)>=2 and bool(d.iloc[-2].signal):
        pr=d.iloc[-2]; psl=float(pr.h4_swing_low); entry=float(r.open)
        if psl < entry:
            tp=entry+2*(entry-psl)
            status="ENTRY"
            detail=f"Prior-session signal | entry ${entry:.2f} | SL ${psl:.2f} | TP ${tp:.2f} | max exit {nth_session(day,10).date()}"
    return {"ticker":"IBM","status":status,"detail":detail,"date":day.date(),
            "close":float(r.close),"extra":f"Daily {r.hi_state}+{r.lo_state} | 4h {r.h4_hi}+{r.h4_lo}"}

# ---------------- Visa frozen strategy ----------------
@st.cache_data(ttl=900)
def v_fetch():
    z=yf.download("V",period="2y",interval="1d",auto_adjust=False,progress=False,threads=False)
    if z.empty: raise RuntimeError("Yahoo returned no V daily data.")
    if isinstance(z.columns,pd.MultiIndex): z.columns=z.columns.get_level_values(0)
    z=z.reset_index()
    z["date"]=pd.to_datetime(z["Date"]).dt.tz_localize(None)
    d=pd.DataFrame({"date":z["date"],"open":z["Open"],"high":z["High"],"low":z["Low"],
                    "close":z["Close"],"volume":z["Volume"]}).dropna().sort_values("date").reset_index(drop=True)
    # EXACT frozen v2 definition used in the backtest:
    d["lo20"]=d.low.rolling(20).min().shift(1)
    d["bear"]=d.close<d.open
    d["sup_touch"]=(d.low-d.lo20).abs()/d.close < .0075
    d["signal"]=d.bear & d.sup_touch
    pc=d.close.shift(1)
    tr=pd.concat([(d.high-d.low),(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr14"]=tr.rolling(14).mean()
    return d

def scan_v():
    d=completed_today_cutoff(v_fetch())
    if d.empty: raise RuntimeError("No completed V session.")
    r=d.iloc[-1]; day=pd.Timestamp(r.date).normalize()
    status="NO SIGNAL"; detail="No frozen V bear_sup_touch pattern."
    if bool(r.signal) and pd.notna(r.atr14):
        future=nyse_sessions(day+pd.Timedelta(days=1),day+pd.Timedelta(days=20))
        ed=future[0] if len(future) else None
        dist=2*float(r.atr14)
        status="SIGNAL"
        detail=f"LONG next open | SL = entry − ${dist:.2f} | TP = entry + ${2*dist:.2f}"
        if ed is not None: detail += f" | expected entry {ed.date()} | max exit {nth_session(ed,3).date()}"
    elif len(d)>=2 and bool(d.iloc[-2].signal) and pd.notna(d.iloc[-2].atr14):
        pr=d.iloc[-2]; entry=float(r.open); dist=2*float(pr.atr14)
        status="ENTRY"
        detail=f"Prior-session signal | entry ${entry:.2f} | SL ${entry-dist:.2f} | TP ${entry+2*dist:.2f} | max exit {nth_session(day,3).date()}"
    return {"ticker":"V","status":status,"detail":detail,"date":day.date(),
            "close":float(r.close),"extra":f"20d support ${float(r.lo20):.2f} | ATR14 ${float(r.atr14):.2f}"}

# ---------------- WFC frozen earnings strategy ----------------
@st.cache_data(ttl=3600)
def wfc_fetch():
    t=yf.Ticker("WFC")
    cal=t.calendar
    hist=t.get_earnings_dates(limit=16)
    px=yf.download("WFC",period="3mo",interval="1d",auto_adjust=False,progress=False,threads=False)
    if isinstance(px.columns,pd.MultiIndex): px.columns=px.columns.get_level_values(0)
    return cal,hist,px

def scan_wfc():
    cal,hist,px=wfc_fetch()
    today=pd.Timestamp(datetime.now(NY).date())
    dates=[]
    if hist is not None and len(hist):
        z=pd.to_datetime(hist.index)
        try: z=z.tz_localize(None)
        except TypeError: z=z.tz_convert(None)
        dates += list(z)
    if isinstance(cal,dict):
        vals=cal.get("Earnings Date",[])
        if not isinstance(vals,(list,tuple)): vals=[vals]
        for v in vals:
            if v is not None:
                q=pd.Timestamp(v)
                if q.tzinfo: q=q.tz_localize(None)
                dates.append(q)
    dates=sorted(set(dates))
    up=[d for d in dates if d.normalize()>=today]
    past=[d for d in dates if d.normalize()<today]
    next_e=up[0] if up else None; last=past[-1] if past else None

    close=np.nan; atr=np.nan
    if px is not None and len(px)>20:
        prev=px["Close"].shift(1)
        tr=pd.concat([(px["High"]-px["Low"]),(px["High"]-prev).abs(),(px["Low"]-prev).abs()],axis=1).max(axis=1)
        atr=float(tr.rolling(14).mean().iloc[-1]); close=float(px["Close"].iloc[-1])

    status="NO SIGNAL"; detail="No current WFC earnings entry."
    # Yahoo calendar date alone does not reliably encode before-open vs after-close timing.
    if last is not None and (today-last.normalize()).days<=4:
        status="VERIFY"
        detail=f"Earnings reported {last.date()} — verify release time; valid entry is the first regular-session open AFTER the release."
    elif next_e is not None:
        detail=f"Next reported earnings date: {next_e.date()} ({(next_e.normalize()-today).days} days)"
    return {"ticker":"WFC","status":status,"detail":detail,"date":today.date(),
            "close":close,"extra":(f"ATR14 ${atr:.2f}" if np.isfinite(atr) else "ATR unavailable")}

# ---------------- UI ----------------
st.title("Frozen Signal Scanner")
st.caption("IBM + WFC + V — only validated frozen strategies. New tickers can be added as they survive the full test process.")

results=[]
for name,fn in [("IBM",scan_ibm),("WFC",scan_wfc),("V",scan_v)]:
    try:
        results.append(fn())
    except Exception as e:
        results.append({"ticker":name,"status":"DATA ERROR","detail":str(e),"date":"—","close":np.nan,"extra":""})

st.subheader("Scanner")
for r in results:
    icon={"SIGNAL":"🟢","ENTRY":"🟢","VERIFY":"🟡","NO SIGNAL":"⚪","DATA ERROR":"🔴"}.get(r["status"],"⚪")
    price=f"${r['close']:.2f}" if np.isfinite(r["close"]) else "—"
    with st.container(border=True):
        c1,c2,c3=st.columns([1,1.2,1])
        c1.markdown(f"### {r['ticker']}")
        c2.markdown(f"**{icon} {r['status']}**")
        c3.metric("Latest close",price)
        st.write(r["detail"])
        if r["extra"]: st.caption(r["extra"])

st.divider()
tabs=st.tabs(["IBM","WFC","V"])
with tabs[0]:
    st.markdown("### IBM frozen rule")
    st.write("Daily LH+LL + synthetic 4h HH+HL → LONG next open → latest confirmed 4h swing-low SL → 2R TP → max 10 sessions.")
with tabs[1]:
    st.markdown("### WFC frozen rule")
    st.write("Every confirmed WFC earnings release → LONG at the next regular-session open after the release → SL = entry − 2×ATR(14) → TP = 2R → max 5 sessions.")
    st.warning("Yahoo's earnings calendar may not reliably distinguish before-open from after-close. Verify the actual release timing before entering.")
with tabs[2]:
    st.markdown("### V frozen rule")
    st.write("Bearish daily candle whose low is within 0.75% of the prior 20-session low → LONG next open → SL = entry − 2×ATR(14) → TP = 2R → max 3 sessions.")

st.caption("Cloud data: Yahoo Finance. Historical validation used IBKR data, so cloud signals should be treated as monitoring signals and checked for provider differences.")
