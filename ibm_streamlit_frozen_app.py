import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import datetime, time
from zoneinfo import ZoneInfo

st.set_page_config(page_title="IBM Frozen Signal", page_icon="📈", layout="centered")

TICKER="IBM"
MAX_HOLD=10
R_TARGET=2.0
COST=0.001
NY=ZoneInfo("America/New_York")

def structure(raw):
    x=raw.copy().reset_index(drop=True)
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

def daily(h):
    x=h.copy(); x["day"]=x.date.dt.normalize()
    return x.groupby("day",as_index=False).agg(date=("day","first"),open=("open","first"),
        high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"))

def fourhour(h):
    # EXACT frozen definition: sequential RTH hourly bars; first 4, then remainder of session.
    x=h.copy(); x["day"]=x.date.dt.normalize()
    x["slot"]=x.groupby("day").cumcount()//4
    return x.groupby(["day","slot"],as_index=False).agg(date=("date","last"),open=("open","first"),
        high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum")).drop(columns=["slot"])

@st.cache_data(ttl=900)
def fetch():
    z=yf.download(TICKER,period="729d",interval="1h",auto_adjust=False,
                  prepost=False,progress=False,threads=False)
    if z.empty: raise RuntimeError("Yahoo returned no IBM hourly data.")
    if isinstance(z.columns,pd.MultiIndex): z.columns=z.columns.get_level_values(0)
    z=z.reset_index()
    dc="Datetime" if "Datetime" in z.columns else "Date"
    dt=pd.to_datetime(z[dc],utc=True).dt.tz_convert(NY)
    z=pd.DataFrame({"date":dt.dt.tz_localize(None),"open":z["Open"],"high":z["High"],
                    "low":z["Low"],"close":z["Close"],"volume":z["Volume"]})
    z=z.dropna().sort_values("date")
    # Explicit RTH safety filter.
    tm=z.date.dt.time
    z=z[(tm>=time(9,30))&(tm<time(16,0))].reset_index(drop=True)
    return z

def build(h):
    d=structure(daily(h)); q=structure(fourhour(h))
    q["day"]=q.date.dt.normalize()
    ql=q.groupby("day").tail(1).set_index("day")
    d["day"]=d.date.dt.normalize()
    d["h4_hi"]=d.day.map(ql.hi_state); d["h4_lo"]=d.day.map(ql.lo_state)
    d["h4_bull"]=d.day.map(ql.bull_struct).fillna(False).astype(bool)
    d["h4_swing_low"]=d.day.map(ql.last_lo)
    d["signal"]=d.bear_struct & d.h4_bull
    return d

def nyse_sessions(start,end):
    cal=mcal.get_calendar("NYSE")
    return cal.schedule(start_date=start,end_date=end).index.tz_localize(None).normalize()

def tenth_session_after(entry_day):
    ss=nyse_sessions(entry_day, entry_day+pd.Timedelta(days=30))
    ss=ss[ss>=pd.Timestamp(entry_day).normalize()]
    return ss[MAX_HOLD-1] if len(ss)>=MAX_HOLD else None

def closed_trades(h,d):
    rows=[]; free=pd.Timestamp.min
    sigs=list(d.index[d.signal])
    for i in sigs:
        if i+1>=len(d): continue
        sigday=pd.Timestamp(d.date.iloc[i]).normalize()
        entryday=pd.Timestamp(d.date.iloc[i+1]).normalize()
        if entryday<free: continue
        sl=float(d.h4_swing_low.iloc[i]) if pd.notna(d.h4_swing_low.iloc[i]) else np.nan
        entry=float(d.open.iloc[i+1])
        if not np.isfinite(sl) or sl>=entry: continue
        risk=entry-sl; tp=entry+R_TARGET*risk
        exit_i=min(i+MAX_HOLD,len(d)-1)
        planned=pd.Timestamp(d.date.iloc[exit_i]).normalize()
        # Don't call a still-open/incomplete max-hold trade closed.
        hh=h[(h.date.dt.normalize()>=entryday)&(h.date.dt.normalize()<=planned)]
        ep=None; ed=None; why=None
        for _,b in hh.iterrows():
            hs=b.low<=sl; ht=b.high>=tp
            if hs and ht: ep=sl; ed=b.date; why="SL*"; break
            if hs: ep=sl; ed=b.date; why="SL"; break
            if ht: ep=tp; ed=b.date; why="TP"; break
        if ep is None:
            if exit_i>=len(d)-1 and pd.Timestamp(d.date.iloc[-1]).normalize()<planned:
                continue
            ep=float(d.close.iloc[exit_i]); ed=d.date.iloc[exit_i]; why="TIME"
        net=ep/entry-1-COST
        rows.append(dict(signal=sigday.date(),entry_date=entryday.date(),entry=entry,
                         stop=sl,target=tp,exit_date=pd.Timestamp(ed).date(),exit=ep,
                         result=why,return_pct=100*net))
        free=pd.Timestamp(ed).normalize()+pd.Timedelta(days=1)
    return pd.DataFrame(rows)

st.title("IBM — Frozen Signal")
st.caption("Daily LH+LL + 4h HH+HL → LONG next open → 4h swing-low SL → 2R TP → max 10 sessions")

try:
    h=fetch(); d=build(h)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

now=datetime.now(NY)
# Use only completed US sessions. During market hours, remove today's partial day.
today=pd.Timestamp(now.date())
if now.time()<time(16,5):
    completed=d[d.date<today].copy()
else:
    completed=d.copy()
if completed.empty:
    st.error("No completed session available."); st.stop()

r=completed.iloc[-1]; idx=completed.index[-1]
day=pd.Timestamp(r.date).normalize()
close=float(r.close)
signal=bool(r.signal)
sl=float(r.h4_swing_low) if pd.notna(r.h4_swing_low) else np.nan

c1,c2=st.columns(2)
c1.metric("Last completed session",str(day.date()))
c2.metric("IBM close",f"${close:.2f}")
st.write(f"**Daily:** {r.hi_state}+{r.lo_state}  |  **4h:** {r.h4_hi}+{r.h4_lo}")

# Determine whether yesterday/last prior completed session signaled and today's open is known.
prior_signal=None
if len(completed)>=2:
    pr=completed.iloc[-2]
    if bool(pr.signal):
        prior_signal=pr

if signal and np.isfinite(sl):
    st.success("SIGNAL CONFIRMED")
    st.write(f"**Action:** LONG at next session open")
    st.write(f"**Exact stop loss:** ${sl:.2f}")
    st.write("**Take profit:** entry + 2 × (entry − stop). Exact TP becomes known at the next session open.")
    # Expected next exchange session + exact max hold date based on NYSE calendar.
    future=nyse_sessions(day+pd.Timedelta(days=1),day+pd.Timedelta(days=45))
    if len(future):
        entry_date=future[0]
        maxdate=tenth_session_after(entry_date)
        st.write(f"**Expected entry date:** {entry_date.date()}")
        if maxdate is not None: st.write(f"**Max-hold exit:** close on {maxdate.date()} if neither SL nor TP has hit")
    indicative=close+2*(close-sl)
    st.caption(f"Indicative TP using today's close as entry: ${indicative:.2f} — not the executable TP.")
    st.markdown("### 🟢 TAKE ACTION NOW — signal confirmed after market close; prepare LONG for next open.")
elif prior_signal is not None and day>pd.Timestamp(prior_signal.date).normalize():
    psl=float(prior_signal.h4_swing_low)
    entry=float(r.open)
    if psl<entry:
        tp=entry+2*(entry-psl)
        maxdate=tenth_session_after(day)
        st.success("PREVIOUS SESSION SIGNAL — ENTRY LEVELS NOW KNOWN")
        st.write(f"**Entry (today's regular-session open):** ${entry:.2f}")
        st.write(f"**SL:** ${psl:.2f}")
        st.write(f"**TP (2R):** ${tp:.2f}")
        if maxdate is not None: st.write(f"**Max-hold exit:** close on {maxdate.date()}")
        st.markdown("### 🟢 TAKE ACTION — frozen signal triggered on the prior session.")
else:
    forming=bool(r.bear_struct) ^ bool(r.h4_bull)
    if forming:
        st.warning("🟡 WATCH — one side of the setup is present, but NO TRADE unless both confirm after close.")
    else:
        st.info("⚪ NO PATTERN FOUND — no action required.")

st.divider()
st.subheader("Latest 5 closed historical trades")
tr=closed_trades(h,completed)
if tr.empty:
    st.write("No closed trades in available cloud-data history.")
else:
    show=tr.tail(5).copy()
    for c in ["entry","stop","target","exit"]: show[c]=show[c].map(lambda x:f"${x:.2f}")
    show["return_pct"]=show["return_pct"].map(lambda x:f"{x:+.2f}%")
    show=show.rename(columns={"signal":"Signal","entry_date":"Entry date","entry":"Entry",
        "stop":"SL","target":"TP","exit_date":"Exit date","exit":"Exit",
        "result":"Result","return_pct":"Return"})
    st.dataframe(show,use_container_width=True,hide_index=True)

st.caption("Cloud data: Yahoo Finance via yfinance, 1h regular-session bars. Strategy rules are frozen; data provider differs from the IBKR backtest.")
