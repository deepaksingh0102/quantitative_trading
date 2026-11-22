"""Quant Panel — SPY & QQQ (display only).

Chan, Quantitative Trading, graded Nov 19, 2026 before building: the
z-score fades, the regime label and the SPY-QQQ spread z-score were
graded on his data and NONE of them has an edge as a signal, so this
app has no signals, no verdict and no Buy/Sell. It shows the same
numbers live, as context:

  * z-score vs the 20-bar mean and z-score vs session VWAP (5-min
    bars, hindsight-free: the mean/std use only bars closed before
    the signal bar — exactly the graded definition).
  * the morning regime label (lag-1 autocorrelation of 5-min log
    returns, 09:30-10:30; mean-reverting / trending / random walk).
  * the SPY-QQQ spread z-score (log QQQ - log SPY, rolling 20 bars).
  * the Kelly sizer, fed by the LAST GRADED expectancy (Nov 19) —
    it reads NO SIZE until a new grade finds a positive edge.

Grade results (Nov 19, 2026 — the numbers the Kelly panel quotes):
  A1 z vs 20-bar mean, QQQ 5-min, 41 sessions Sep 4-Nov 2 (SIP file):
    +30m n=87 hit 52.9% avg -0.9 bps gross / -2.9 net; in-sample
    +30m hit 61.4% faded to 44.2% out-of-sample. No edge.
  A2 z vs session VWAP, same file: +30m n=94 hit 60.6% avg +0.3 /
    -1.7 net — hit rate without edge. No edge.
  B regime label: morning label matched the full-session label only
    56% of sessions; fades lost on average in every regime.
  C spread z-score, Yahoo 5-min, 60 sessions Aug 27-Nov 19: +30m
    n=137 hit 59.1% avg +1.8 bps gross / -2.2 net — held out of
    sample, but +1-2 bps gross vs 4 bps two-leg costs. No edge.
  Kelly on A1 +30m: p=0.529, avg win 10.6 bps, avg loss 14.2 bps,
    edge -1.05 bps/trade -> Kelly f <= 0: no bet.

Data: Alpaca SIP 1-min bars (same Secrets as the stock-analyzer and
the price-action charting app — keys live in Streamlit Secrets, never
in the file). Live tracking auto-refreshes during US market hours and
goes static after the close, like the other apps.

Bulkowski, Encyclopedia of Chart Patterns, graded Nov 20, 2026 before
adding anything: five rules plus the throwback follow-up, on QQQ 5-min
SIP 41 sessions Sep 4-Nov 2 and SPY+QQQ Yahoo 5-min 60 sessions
Aug 27-Nov 19, signals 10:00 ET onward, one per session per rule,
+15/+30/+60m, 2 bps/trade, split-sample. NO RULE HAS AN EDGE.
  * 12-bar range breakout: n=41, +30m +2.4 bps net — halves disagree
    (in-sample -2.1, out-of-sample +6.7). No edge.
  * Breakout + flagpole >=1%: 0 of 13 pass; max flagpole 0.46%.
    Untestable — the book's threshold is 2-6x the typical QQQ hourly
    move. Breakout + volume: heavy n=8 -4.3 bps gross, light n=11
    +4.8 bps gross at +30m — direction matches the book (light wins)
    but n is far too small. Inconclusive.
  * Failed breakout, trade the reversal: n=41, +30m -6.9 bps net,
    out-of-sample -11.0 bps gross. No edge.
  * Partial decline/rise as a filter: 1 occurrence in 13. Inconclusive.
  * Fade the overnight gap: n=66, +30m -3.1 bps net; fills 25% by
    close vs the book's 89-93% within a week. No edge.
  * Follow the overnight gap: n=24, +60m +10.2 bps net — the largest
    positive number in the grade, but n=24 and the short side (n=10)
    carries it. Inconclusive, retest candidate, not a signal.
  * Measured-move second leg: n=1 (funnel 41 sessions -> 5 drives ->
    2 retraces -> 1 trigger). Inconclusive.
  * Throwback re-entry: +30m +4.0 bps net, in-sample -0.7. No edge;
    +15m is the second retest candidate.
Two descriptive facts are real and stay in the panel as context, not
trades: 76% of 12-bar breakouts return to the breakout level within
12 bars, and after a failed breakout price re-takes the breakout
extreme 73% of the time before reaching the reversal target — the trap
usually isn't a trap on 5-min QQQ, the original breakout resumes.
Rectangle measure target is reached first only 15% (book: 80-85% on
daily bars). Kelly stays NO SIZE for every Bulkowski rule.
"""
import hashlib
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ET = ZoneInfo("America/New_York")
ALPACA_BASE = "https://data.alpaca.markets/v2"
SYMBOLS = ("SPY", "QQQ")
LOAD_DAYS = 7            # calendar days of 1-min bars (covers ~5 sessions)
BAR_MIN = 5              # the graded bar size
Z_WINDOW = 20            # bars for the rolling mean/std
Z_THRESH = 2.0           # extreme threshold
REGIME_WINDOW_BARS = 12  # 09:30-10:30 on 5-min bars
BUILD_TAG = "2026-11-22-unified-patterns-v14"

# ---------------------------------------------------------------- build tag
def _build_hash() -> str:
    try:
        return hashlib.md5(open(__file__, "rb").read()).hexdigest()[:6]
    except Exception:
        return "??????"


BUILD_HASH = _build_hash()

# ---------------------------------------------------------------- Alpaca
def _secrets() -> tuple[str, str, str]:
    def _get(name, default=""):
        try:
            v = st.secrets.get(name, default)
        except Exception:
            v = default
        import os
        return str(os.environ.get(name, v) or default)
    return (_get("APCA_API_KEY_ID"), _get("APCA_API_SECRET_KEY"),
            _get("ALPACA_FEED", "sip") or "sip")


def alpaca_get(path: str, params: dict, tries: int = 4) -> dict:
    key, secret, _ = _secrets()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    delay = 1.0
    for _ in range(tries):
        r = requests.get(f"{ALPACA_BASE}{path}", params=params,
                         headers=headers, timeout=30)
        if r.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 2, 16.0)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def _bucket(live: bool, refresh_secs: int) -> int:
    base = int(time.time() // refresh_secs) if live else 0
    try:
        nonce = st.session_state.get("refresh_nonce", 0)
    except Exception:
        nonce = 0
    return base + nonce


@st.cache_data(show_spinner=False)
def fetch_bars_1m(symbol: str, bucket: int = 0,
                  days: int = LOAD_DAYS) -> pd.DataFrame:
    """Last `days` calendar days of 1-minute bars, regular hours only
    (09:30-16:00 ET), paginated. Same fetch as the price-action
    charting app."""
    _, _, feed = _secrets()
    start = (datetime.now(ET) - timedelta(days=days)).strftime(
        "%Y-%m-%dT00:00:00Z")
    rows, token = [], None
    while True:
        params = {"timeframe": "1Min", "start": start, "limit": 10000,
                  "feed": feed, "sort": "asc", "adjustment": "split"}
        if token:
            params["page_token"] = token
        data = alpaca_get(f"/stocks/{symbol}/bars", params)
        rows.extend(data.get("bars") or [])
        token = data.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={
        "t": "ts", "o": "Open", "h": "High", "l": "Low", "c": "Close",
        "v": "Volume"})
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df["ts"], utc=True)).tz_convert(ET)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    df = df.sort_index()
    mins = df.index.hour * 60 + df.index.minute
    df = df[(mins >= 570) & (mins < 960)]          # 09:30 <= t < 16:00 ET
    return df[~df.index.duplicated(keep="last")]


# ---------------------------------------------------------------- pure math
def resample_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.resample("5min", origin="start_day").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last",
         "Volume": "sum"}).dropna(subset=["Close"])


def session_vwap(df5: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP on 5-min bars (the resampled
    approximation, same as the charting app's add_indicators)."""
    tp = (df5["High"] + df5["Low"] + df5["Close"]) / 3.0
    dates = df5.index.date
    pv = (tp * df5["Volume"]).groupby(dates).cumsum()
    vv = df5["Volume"].groupby(dates).cumsum()
    return pv / vv.replace(0, np.nan)


def zscore(s: pd.Series, n: int = Z_WINDOW) -> pd.Series:
    """Hindsight-free z-score, exactly the graded definition:
    (x - mean of the prior n values) / std of the prior n values.
    The first n bars are NaN — there is no look-back yet."""
    m = s.shift(1).rolling(n).mean()
    sd = s.shift(1).rolling(n).std()
    return (s - m) / sd


def z_extremes(z: pd.Series, thresh: float = Z_THRESH) -> list[dict]:
    """First fresh crossings of |z| >= thresh, re-arming only after z
    returns inside +/-0.5 — the graded dedupe. Returns the events
    most recent first: time, side, z value."""
    out = []
    vals = z.to_numpy(dtype=float)
    armed = True
    for i, v in enumerate(vals):
        if np.isnan(v):
            continue
        if armed and v <= -thresh:
            out.append({"time": z.index[i], "side": "long",
                        "z": float(v)})
            armed = False
        elif armed and v >= thresh:
            out.append({"time": z.index[i], "side": "short",
                        "z": float(v)})
            armed = False
        elif not armed and abs(v) < 0.5:
            armed = True
    out.reverse()
    return out


def regime_label(df5: pd.DataFrame) -> dict:
    """Morning regime from the 09:30-10:30 window (first 12 5-min
    bars): lag-1 autocorrelation of log returns. < -0.1 =
    mean-reverting, > +0.1 = trending, else random walk. Before
    10:30 the label is computed on the bars so far and marked
    forming. The graded stability check: the morning label matched
    the full-session label in only 56% of the 41 graded sessions —
    a hint, not a classification."""
    morn = df5.iloc[:REGIME_WINDOW_BARS]
    n_bars = len(morn)
    if n_bars < 6:
        return {"label": "forming", "ac": None, "n_bars": n_bars,
                "complete": False}
    lr = np.diff(np.log(morn["Close"].to_numpy(dtype=float)))
    if len(lr) < 4:
        return {"label": "forming", "ac": None, "n_bars": n_bars,
                "complete": False}
    ac = float(np.corrcoef(lr[:-1], lr[1:])[0, 1])
    label = ("mean-reverting" if ac < -0.1 else
             "trending" if ac > 0.1 else "random walk")
    return {"label": label, "ac": ac, "n_bars": n_bars,
            "complete": n_bars >= REGIME_WINDOW_BARS}


def spread_z(spy5: pd.DataFrame, qqq5: pd.DataFrame) -> pd.DataFrame:
    """Aligned 5-min closes; spread = log(QQQ) - log(SPY); z from the
    prior 20 spread values, within the session (reset each day, as
    graded). Returns a frame with spread and z columns."""
    j = pd.DataFrame({"spy": spy5["Close"], "qqq": qqq5["Close"]})
    j = j.dropna()
    j["spread"] = np.log(j["qqq"]) - np.log(j["spy"])
    j["date"] = j.index.date
    j["z"] = np.nan
    for _d, g in j.groupby("date"):
        j.loc[g.index, "z"] = zscore(g["spread"]).to_numpy()
    return j


def symbol_read(df5: pd.DataFrame, prev_close: float | None) -> dict:
    """Everything the panel shows for one symbol: current z-scores,
    regime, extremes, price vs VWAP."""
    vwap = session_vwap(df5)
    z_mean = zscore(df5["Close"])
    dev = df5["Close"] - vwap
    z_vwap = zscore(dev)
    last = df5.iloc[-1]
    last_close = float(last["Close"])
    out = {
        "last_close": last_close,
        "last_bar": df5.index[-1],
        "vwap": float(vwap.iloc[-1]),
        "z_mean": float(z_mean.iloc[-1])
        if pd.notna(z_mean.iloc[-1]) else None,
        "z_vwap": float(z_vwap.iloc[-1])
        if pd.notna(z_vwap.iloc[-1]) else None,
        "z_mean_series": z_mean,
        "z_vwap_series": z_vwap,
        "regime": regime_label(df5),
        "extremes_mean": z_extremes(z_mean),
        "extremes_vwap": z_extremes(z_vwap),
        "chg_pct": ((last_close / prev_close - 1) * 100
                    if prev_close else None),
    }
    return out


# ---------------------------------------------------------------- graded Kelly
GRADED = {
    "date": "Nov 19, 2026",
    "a1": {"name": "z vs 20-bar mean (QQQ 5-min, 41 sessions)",
           "horizon": "+30m", "n": 87, "hit": 52.9,
           "avg_gross_bps": -0.9, "avg_net_bps": -2.9,
           "p": 0.529, "avg_win_bps": 10.6, "avg_loss_bps": 14.2,
           "edge_bps": -1.05},
    "a2": {"name": "z vs session VWAP (QQQ 5-min, 41 sessions)",
           "horizon": "+30m", "n": 94, "hit": 60.6,
           "avg_gross_bps": 0.3, "avg_net_bps": -1.7,
           "p": None, "avg_win_bps": None, "avg_loss_bps": None,
           "edge_bps": None},
    "c": {"name": "SPY-QQQ spread z (Yahoo 5-min, 60 sessions)",
          "horizon": "+30m", "n": 137, "hit": 59.1,
          "avg_gross_bps": 1.8, "avg_net_bps": -2.2,
          "p": None, "avg_win_bps": None, "avg_loss_bps": None,
          "edge_bps": None},
}


def kelly_line() -> str:
    """The Kelly sizer, fed by the last graded expectancy (A1 +30m).
    f <= 0 means no bet — the panel sizes nothing until a new grade
    finds a positive edge. The edge is quoted exactly as graded
    (recomputing it from the rounded win rate would give -1.08)."""
    g = GRADED["a1"]
    return (f"Kelly on the graded expectancy (A1 +30m, n={g['n']}): "
            f"win rate {g['p']:.1%}, avg win {g['avg_win_bps']:.1f} bps, "
            f"avg loss {g['avg_loss_bps']:.1f} bps, edge "
            f"{g['edge_bps']:+.2f} bps/trade — Kelly f ≤ 0, **no size**. "
            f"The panel sizes nothing until a new grade finds a "
            f"positive edge.")


# ---------------------------------------------------------------- charts
def z_chart(sym: str, read: dict) -> go.Figure:
    """Both z series for one symbol, with the +/-2 bands. The title
    leads with the current values."""
    zm, zv = read["z_mean_series"], read["z_vwap_series"]
    zmt = f"{read['z_mean']:+.2f}" if read["z_mean"] is not None else "—"
    zvt = f"{read['z_vwap']:+.2f}" if read["z_vwap"] is not None else "—"
    fig = go.Figure()
    fig.add_hline(y=2.0, line_dash="dash", line_color="#9ca3af",
                  annotation_text="+2", annotation_position="right")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="#9ca3af",
                  annotation_text="−2", annotation_position="right")
    fig.add_hline(y=0.0, line_color="#d1d5db")
    fig.add_trace(go.Scatter(x=zm.index, y=zm, name="z vs 20-bar mean",
                             line=dict(color="#2563eb", width=2)))
    fig.add_trace(go.Scatter(x=zv.index, y=zv, name="z vs session VWAP",
                             line=dict(color="#d97706", width=2)))
    fig.update_layout(
        title=f"{sym} — z vs mean {zmt} · z vs VWAP {zvt}",
        xaxis_title="", yaxis_title="z-score",
        height=300, margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(range=[zm.index[0].replace(hour=9, minute=25),
                          zm.index[0].replace(hour=16, minute=0)]))
    return fig


def spread_chart(sp: pd.DataFrame) -> go.Figure:
    cur = sp["z"].iloc[-1] if len(sp) else None
    curt = f"{cur:+.2f}" if cur is not None and pd.notna(cur) else "—"
    fig = go.Figure()
    fig.add_hline(y=2.0, line_dash="dash", line_color="#9ca3af",
                  annotation_text="+2", annotation_position="right")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="#9ca3af",
                  annotation_text="−2", annotation_position="right")
    fig.add_hline(y=0.0, line_color="#d1d5db")
    fig.add_trace(go.Scatter(x=sp.index, y=sp["z"],
                             name="spread z", line=dict(color="#7c3aed",
                                                        width=2)))
    fig.update_layout(
        title=f"SPY–QQQ spread z — now {curt}",
        xaxis_title="", yaxis_title="z-score",
        height=300, margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(range=[sp.index[0].replace(hour=9, minute=25),
                          sp.index[0].replace(hour=16, minute=0)]))
    return fig


# ---------------------------------------------------------------- market hours
def in_market_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 570 <= mins < 960



# ------------------------------------------------------ Aziz price-action lab
# Educational, ungraded hypotheses. No automatic orders or trade recommendations.
def completed_5m(df1: pd.DataFrame, asof=None) -> pd.DataFrame:
    """Exclude the currently forming five-minute candle during live trading."""
    d = resample_5min(df1)
    if d.empty:
        return d
    now = asof or datetime.now(ET)
    # Historical sessions are fully complete. For today's session only, require
    # the 5-minute interval to have ended, even if a data vendor is delayed.
    if d.index[-1].date() == now.date():
        d = d[d.index + pd.Timedelta(minutes=5) <= pd.Timestamp(now)]
    return d


def completed_1m(df1: pd.DataFrame, asof=None) -> pd.DataFrame:
    """Use only completed 1-minute Alpaca candles, including in live mode."""
    if df1.empty:
        return df1.copy()
    now = pd.Timestamp(asof or datetime.now(ET))
    if now.tzinfo is None:
        now = now.tz_localize(ET)
    out = df1.sort_index().copy()
    if out.index[-1].date() == now.date():
        out = out[out.index + pd.Timedelta(minutes=1) <= now]
    return out


def price_action_features(df5: pd.DataFrame, prev: pd.DataFrame | None = None,
                          opening_minutes: int = 15, bar_minutes: int = 5) -> pd.DataFrame:
    """All indicators use only information known at each completed bar."""
    x = df5.copy()
    if x.empty:
        return x
    x['EMA9'] = x.Close.ewm(span=9, adjust=False).mean()
    x['EMA20'] = x.Close.ewm(span=20, adjust=False).mean()
    x['VWAP'] = session_vwap(x)
    # Previous completed five-minute bars only; never use future volume.
    x['RVOL20'] = x.Volume / x.Volume.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)
    x['Range'] = x.High - x.Low
    x['Body'] = (x.Close - x.Open).abs()
    x['UpperWick'] = x.High - x[['Open','Close']].max(axis=1)
    x['LowerWick'] = x[['Open','Close']].min(axis=1) - x.Low
    x['Doji'] = (x.Range > 0) & (x.Body / x.Range.replace(0, np.nan) <= .15)
    x['BullEngulf'] = ((x.Close > x.Open) & (x.Close.shift(1) < x.Open.shift(1)) &
                       (x.Open <= x.Close.shift(1)) & (x.Close >= x.Open.shift(1)))
    x['BearEngulf'] = ((x.Close < x.Open) & (x.Close.shift(1) > x.Open.shift(1)) &
                       (x.Open >= x.Close.shift(1)) & (x.Close <= x.Open.shift(1)))
    x['HHHL'] = (x.High > x.High.shift(1)) & (x.Low > x.Low.shift(1))
    x['LLLH'] = (x.High < x.High.shift(1)) & (x.Low < x.Low.shift(1))
    x['BullReject'] = (x.Range > 0) & (x.LowerWick >= 2*x.Body) & (x.Close >= x.Open)
    x['BearReject'] = (x.Range > 0) & (x.UpperWick >= 2*x.Body) & (x.Close <= x.Open)
    n = max(1, opening_minutes // bar_minutes)
    x['ORHigh'] = np.nan
    x['ORLow'] = np.nan
    if len(x) >= n:
        # Opening range only becomes available after its last candle closes.
        x.loc[x.index[n-1]:, 'ORHigh'] = float(x.High.iloc[:n].max())
        x.loc[x.index[n-1]:, 'ORLow'] = float(x.Low.iloc[:n].min())
    x['ORBreakUp'] = (x.Close > x.ORHigh) & (x.Close.shift(1) <= x.ORHigh) & (x.index >= x.index[min(n,len(x)-1)])
    x['ORBreakDown'] = (x.Close < x.ORLow) & (x.Close.shift(1) >= x.ORLow) & (x.index >= x.index[min(n,len(x)-1)])
    x['VWAPCrossUp'] = (x.Close > x.VWAP) & (x.Close.shift(1) <= x.VWAP.shift(1))
    x['VWAPCrossDown'] = (x.Close < x.VWAP) & (x.Close.shift(1) >= x.VWAP.shift(1))
    if prev is not None and not prev.empty:
        x.attrs['prior_levels'] = {'PDH':float(prev.High.max()),
                                   'PDL':float(prev.Low.min()),
                                   'PDC':float(prev.Close.iloc[-1])}
    return x


def aziz_events(x: pd.DataFrame, min_rvol: float = 1.5) -> pd.DataFrame:
    """Observation log, not validated entries; no lookahead."""
    rows=[]
    if x.empty:
        return pd.DataFrame(columns=['Time (ET)','Observation','Direction','RVOL','Context'])
    for ts, r in x.iterrows():
        flags=[('ORBreakUp','15m opening-range break','up'),
               ('ORBreakDown','15m opening-range break','down'),
               ('VWAPCrossUp','VWAP cross','up'),
               ('VWAPCrossDown','VWAP cross','down'),
               ('BullEngulf','Bullish engulfing','up'),
               ('BearEngulf','Bearish engulfing','down')]
        for col, name, direction in flags:
            if bool(r[col]):
                vol = r['RVOL20']
                rows.append({'Time (ET)':ts.strftime('%H:%M'),
                             'Observation':name,'Direction':direction,
                             'RVOL':round(float(vol),2) if pd.notna(vol) else None,
                             'Context':('elevated volume' if pd.notna(vol) and vol >= min_rvol
                                        else 'volume unconfirmed')})
    return pd.DataFrame(rows[-25:][::-1],columns=['Time (ET)','Observation','Direction','RVOL','Context'])


def aziz_chart(sym: str, x: pd.DataFrame, show_events: bool = True, event_count: int = 6) -> go.Figure:
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=x.index,open=x.Open,high=x.High,low=x.Low,
                                 close=x.Close,name='Price',increasing_line_color='#16a34a',
                                 decreasing_line_color='#dc2626'))
    for name,color in [('EMA9','#eab308'),('EMA20','#2563eb'),('VWAP','#7c3aed')]:
        fig.add_trace(go.Scatter(x=x.index,y=x[name],name=name,
                                 line=dict(color=color,width=1.7)))
    levels=dict(x.attrs.get('prior_levels',{}))
    if x.ORHigh.notna().any():
        levels['OR High']=float(x.ORHigh.dropna().iloc[-1]); levels['OR Low']=float(x.ORLow.dropna().iloc[-1])
    colors={'PDH':'#059669','PDL':'#be123c','PDC':'#64748b','OR High':'#0d9488','OR Low':'#ea580c'}
    for name,price in levels.items():
        fig.add_hline(y=price,line_dash='dot',line_color=colors.get(name,'#999'),
                      annotation_text=f'{name} {price:.2f}',annotation_position='right')
    if show_events:
        # Use closed-bar timestamps and mark the relevant candle, rather than
        # drawing large text annotations over the price series.
        markers = []
        definitions = [
            ('ORBreakUp', 'OR break up', 'up'),
            ('ORBreakDown', 'OR break down', 'down'),
            ('VWAPCrossUp', 'VWAP cross up', 'up'),
            ('VWAPCrossDown', 'VWAP cross down', 'down'),
            ('BullEngulf', 'Bull engulfing', 'up'),
            ('BearEngulf', 'Bear engulfing', 'down'),
        ]
        for ts, r in x.iterrows():
            for field, label, direction in definitions:
                if bool(r[field]):
                    markers.append((ts, label, direction, r))
        markers = markers[-event_count:]
        # Separate upward and downward observations. Hover gives the precise
        # pattern, bar time and volume confirmation without cluttering chart.
        for direction, color, symbol, name in [
            ('up', '#059669', 'triangle-up', 'Up observations'),
            ('down', '#dc2626', 'triangle-down', 'Down observations'),
        ]:
            selected = [(ts, label, r) for ts, label, d, r in markers if d == direction]
            if not selected:
                continue
            xx = [ts for ts, _, _ in selected]
            yy = [(float(r.Low) - max(float(r.High-r.Low)*0.24, float(r.Close)*0.0003))
                  if direction == 'up' else
                  (float(r.High) + max(float(r.High-r.Low)*0.24, float(r.Close)*0.0003))
                  for _, _, r in selected]
            hover = []
            for ts, label, r in selected:
                rv = f"{r.RVOL20:.2f}x" if pd.notna(r.RVOL20) else 'not available'
                hover.append(f"{label}<br>{ts.strftime('%H:%M')} ET (bar start)<br>"
                             f"Close ${r.Close:.2f}<br>RVOL {rv}<br>Research only - not a trade signal")
            fig.add_trace(go.Scatter(
                x=xx, y=yy, mode='markers', name=name,
                marker=dict(symbol=symbol, size=13, color=color,
                            line=dict(color='white', width=1)),
                text=hover, hovertemplate='%{text}<extra></extra>',
                showlegend=True,
            ))
    fig.update_layout(
        title=dict(text=f'{sym} - Aziz price action (closed 5m bars)',
                   x=0.01, xanchor='left', y=0.98, yanchor='top'),
        height=520,
        xaxis_rangeslider_visible=False,
        dragmode='zoom',
        margin=dict(l=40, r=115, t=115, b=40),
        legend=dict(orientation='h', x=0, xanchor='left',
                    y=1.07, yanchor='bottom',
                    bgcolor='rgba(255,255,255,0)'),
    )
    return fig


def decision_analysis(x: pd.DataFrame, risk_dollars: float = 100.0,
                      min_rvol: float = 1.2, min_rr: float = 2.0) -> dict:
    """Descriptive, ungraded setup checklist; never a live order or forecast."""
    if len(x) < 21:
        return {'status': 'WAIT - insufficient history', 'reasons': ['Need at least 21 completed five-minute bars.'], 'plan': None}
    r = x.iloc[-1]
    close = float(r.Close)
    vwap = float(r.VWAP)
    ema_bear = r.EMA9 < r.EMA20
    ema_bull = r.EMA9 > r.EMA20
    # Session context uses recent *completed* bars, not a single two-bar pattern.
    look = x.tail(6)
    lower = int((look.High.diff() < 0).sum()) + int((look.Low.diff() < 0).sum())
    higher = int((look.High.diff() > 0).sum()) + int((look.Low.diff() > 0).sum())
    session_bias = ('Bearish' if close < vwap and ema_bear else
                    'Bullish' if close > vwap and ema_bull else 'Mixed')
    local = 'Higher highs/lows' if higher >= lower + 3 else ('Lower highs/lows' if lower >= higher + 3 else 'Mixed')
    rv = float(r.RVOL20) if pd.notna(r.RVOL20) else None
    vol_ok = rv is not None and rv >= min_rvol
    prior = x.attrs.get('prior_levels', {})
    levels = {k: float(v) for k,v in prior.items() if pd.notna(v)}
    if pd.notna(r.ORHigh): levels['OR High'] = float(r.ORHigh)
    if pd.notna(r.ORLow): levels['OR Low'] = float(r.ORLow)
    # No current-bar high/low as a hypothetical target: only known levels.
    supports = sorted([(k,v) for k,v in levels.items() if v < close], key=lambda t: -t[1])
    resistances = sorted([(k,v) for k,v in levels.items() if v > close], key=lambda t: t[1])
    reasons = [f'Session context: {session_bias} (price vs VWAP and EMA alignment).',
               f'Recent six-bar structure: {local}; latest two-bar label is {"HH/HL" if r.HHHL else "LH/LL" if r.LLLH else "mixed"}.',
               f'RVOL {rv:.2f}x vs previous 20 bars; threshold {min_rvol:.2f}x.' if rv is not None else 'RVOL unavailable.']
    status = 'WAIT - confirmation needed'
    side = None
    if session_bias == 'Bearish' and local != 'Higher highs/lows' and vol_ok and r.Close < r.EMA9:
        status, side = 'BEARISH continuation candidate - unvalidated', 'short'
    elif session_bias == 'Bullish' and local != 'Lower highs/lows' and vol_ok and r.Close > r.EMA9:
        status, side = 'BULLISH continuation candidate - unvalidated', 'long'
    elif session_bias == 'Bearish' and local == 'Higher highs/lows':
        reasons.append('Local rebound conflicts with bearish session; reversal needs VWAP/EMA reclaim and volume.')
    if not vol_ok: reasons.append('Volume confirmation absent; do not infer a reliable breakout.')
    plan = None
    if side:
        # Reference stop from *previous* closed candle and current EMA20.
        # No forced targets: require a genuine pre-existing level beyond entry.
        prior_bar = x.iloc[-2]
        stop = max(float(prior_bar.High), float(r.EMA20)) if side == 'short' else min(float(prior_bar.Low), float(r.EMA20))
        risk = stop - close if side == 'short' else close - stop
        candidates = supports if side == 'short' else resistances
        targets = [(k,v) for k,v in candidates if (close-v if side == 'short' else v-close) > 0]
        if risk > 0 and targets:
            target_name, target = targets[0]
            rr = abs(target-close)/risk
            shares = int(risk_dollars/risk)
            plan = {'side': side, 'entry': close, 'stop': stop, 'target': target,
                    'target_name': target_name, 'rr': rr, 'shares': shares,
                    'risk_per_share': risk}
            if rr < min_rr or shares < 1:
                status = 'WAIT - risk/reward or sizing fails'
                reasons.append(f'Nearest known target gives {rr:.2f}R; minimum {min_rr:.2f}R.')
        else:
            status = 'WAIT - no valid nearby stop/target'
            reasons.append('No valid reference stop and pre-existing target level.')
    return {'status': status, 'bias': session_bias, 'local': local,
            'reasons': reasons, 'plan': plan, 'support': supports[:2],
            'resistance': resistances[:2], 'bar_end': x.index[-1] + pd.Timedelta(minutes=5)}


def render_decision(sym: str, x: pd.DataFrame):
    st.markdown('#### Trade Decision Analysis — research checklist')
    with st.expander('Research thresholds / hypothetical risk budget', expanded=False):
        risk_budget = st.number_input('Hypothetical dollars at risk', min_value=1.0, value=100.0,
                                      step=25.0, key=f'dec_risk_{sym}')
        min_rvol = st.number_input('Minimum 5m RVOL', min_value=0.5, max_value=10.0,
                                   value=1.2, step=0.1, key=f'dec_vol_{sym}')
        min_rr = st.number_input('Minimum reward/risk', min_value=0.5, max_value=10.0,
                                 value=2.0, step=0.25, key=f'dec_rr_{sym}')
    d = decision_analysis(x, risk_budget, min_rvol, min_rr)
    end = x.index[-1] + pd.Timedelta(minutes=5)
    st.markdown(f'**{d["status"]}** · Latest completed bar: **{end.strftime("%Y-%m-%d %H:%M ET")}**')
    now = datetime.now(ET)
    if end.date() == now.date() and in_market_hours(now):
        lag = (pd.Timestamp(now) - end).total_seconds() / 60
        if lag > 10:
            st.warning(f'Data may be stale: last completed bar ended {lag:.0f} minutes ago.')
    for reason in d['reasons']:
        st.markdown(f'- {reason}')
    if d.get('support') is not None:
        fmt = lambda items: ', '.join(f'{k} ${v:.2f}' for k,v in items) if items else 'none available'
        st.caption(f'Known supports: {fmt(d["support"])} | Known resistances: {fmt(d["resistance"])}')
    plan = d.get('plan')
    if plan:
        st.markdown(f'**Hypothetical {plan["side"]} reference only:** entry ${plan["entry"]:.2f} '
                    f'· stop ${plan["stop"]:.2f} · target {plan["target_name"]} '
                    f'${plan["target"]:.2f} · reward/risk {plan["rr"]:.2f}R '
                    f'· max {plan["shares"]} whole shares at ${risk_budget:.0f} risk before costs.')
    st.caption('Not a validated trade signal or order. Levels and sizing are illustrative, '
               'exclude spreads, slippage and buying-power constraints. Backtest out of sample before use.')




# -------------------------------------------------------- original Weinstein weekly context
# Original book: 30-week MA for investor stage analysis; 10-week MA for
# trader-oriented trend context. Weekly stages are descriptive, not 5m signals.
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_daily_for_weinstein(symbol: str, bucket: int = 0, days: int = 420) -> pd.DataFrame:
    _, _, feed = _secrets()
    start = (datetime.now(ET) - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
    rows, token = [], None
    while True:
        params = {'timeframe':'1Day', 'start':start, 'limit':10000,
                  'feed':feed, 'sort':'asc', 'adjustment':'split'}
        if token: params['page_token'] = token
        response = alpaca_get(f'/stocks/{symbol}/bars', params)
        rows.extend(response.get('bars') or [])
        token = response.get('next_page_token')
        if not token: break
    if not rows: return pd.DataFrame()
    d = pd.DataFrame(rows).rename(columns={'t':'ts','o':'Open','h':'High','l':'Low',
                                            'c':'Close','v':'Volume'})
    d.index = pd.DatetimeIndex(pd.to_datetime(d['ts'],utc=True)).tz_convert(ET)
    return d[['Open','High','Low','Close','Volume']].astype(float).sort_index()


def original_weekly_stage(daily: pd.DataFrame, now=None) -> dict:
    if daily.empty: return {'stage':'Unavailable','detail':'No daily history'}
    now = pd.Timestamp(now or datetime.now(ET))
    # Do not use the current, incomplete weekly bar: last completed Friday only.
    w = daily.resample('W-FRI', label='right', closed='right').agg({
        'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'
    }).dropna(subset=['Close'])
    cutoff = now.normalize() - pd.Timedelta(days=(now.weekday()-4)%7)
    # At Friday close the week is complete; before that, exclude it.
    if now.weekday()==4 and now.hour < 16: cutoff -= pd.Timedelta(days=7)
    w = w[w.index.tz_localize(None).normalize() <= cutoff.tz_localize(None)]
    if len(w)<36:
        return {'stage':'Insufficient weekly data', 'detail':f'{len(w)} completed weeks; need 36'}
    close=w.Close
    ma30=close.rolling(30).mean()
    ma10=close.rolling(10).mean()
    slope=float((ma30.iloc[-1]/ma30.iloc[-5]-1)*100)
    last=float(close.iloc[-1]); avg=float(ma30.iloc[-1])
    prior_high=float(w.High.iloc[-14:-1].max())
    prior_low=float(w.Low.iloc[-14:-1].min())
    width=(prior_high-prior_low)/last
    if last>avg and slope>0.25 and last>prior_high*0.995:
        stage='Stage 2-like / advancing'
    elif last<avg and slope< -0.25 and last<prior_low*1.005:
        stage='Stage 4-like / declining'
    elif abs(slope)<0.35 and width<0.15:
        stage='Stage 1/3-like / sideways (ambiguous)'
    else:
        stage='Transition / stage uncertain'
    return {'stage':stage,'ma30':avg,'ma10':float(ma10.iloc[-1]),
            'last':last,'week':w.index[-1],
            'detail':f'Completed week {w.index[-1].date()}; close ${last:.2f}; 30-week MA ${avg:.2f} (4-week slope {slope:+.2f}%); 10-week MA ${ma10.iloc[-1]:.2f}. This heuristic is not a canonical Weinstein stage classification.'}


def relative_strength_context(spy5: pd.DataFrame, qqq5: pd.DataFrame) -> dict:
    j=pd.concat([spy5.Close.rename('SPY'),qqq5.Close.rename('QQQ')],axis=1).dropna()
    if len(j)<13: return {'text':'Insufficient synchronized SPY/QQQ bars'}
    ratio=j.QQQ/j.SPY
    r3=float((ratio.iloc[-1]/ratio.iloc[-4]-1)*10000)
    r12=float((ratio.iloc[-1]/ratio.iloc[-13]-1)*10000)
    return {'r3':r3,'r12':r12,
            'text':f'QQQ/SPY relative strength: {r3:+.1f} bps over 15m; {r12:+.1f} bps over 60m. Positive means QQQ outperformed SPY; this is not a directional probability.'}

# -------------------------------------------------------- forward direction research
"""Walk-forward directional research for 5-minute SPY/QQQ bars.
No orders. Probabilities are uncalibrated model estimates, with held-out diagnostics.
Includes Edwards-Magee-inspired causal structural features.
"""

BASE_FEATURES = ['ema_gap','vwap_gap','momentum3','momentum6','rvol','range_pct','structure','or_position','z20','macd_hist']
MAGEE_FEATURES = ['trend_efficiency','swing_position','resistance_distance','support_distance',
                  'breakout_up','breakout_down','failed_up','failed_down',
                  'compression','wick_pressure','volume_confirmation']
WEINSTEIN_FEATURES = ['ws_ma_slope', 'ws_base_width', 'ws_breakout_strength',
                     'ws_pullback_quality', 'ws_overhead_atr', 'ws_volume_on_break']
FEATURES = BASE_FEATURES + MAGEE_FEATURES  # Existing V6 production research unchanged.
WEINSTEIN_COMBINED = FEATURES + WEINSTEIN_FEATURES
WEINSTEIN_ORIGINAL_TEST = WEINSTEIN_COMBINED + ["ws_rs_15m", "ws_rs_60m"]
PRING_FEATURES = ['pring_roc3', 'pring_roc6', 'pring_roc12',
                  'pring_momentum_accel', 'pring_momentum_confirmation',
                  'pring_bear_div', 'pring_bull_div', 'pring_roc6_z']
PRING_COMBINED = FEATURES + PRING_FEATURES
# Short-term KST time-frame row in Pring, 4th ed., Chapter 12, Table 12-2.
# Applied to completed five-minute bars as an EXPERIMENTAL intraday adaptation.
PRING_BOOK_FEATURES = ['pring_kst', 'pring_kst_signal_gap', 'pring_kst_slope',
                       'pring_adx14', 'pring_di_spread', 'pring_roc_price_confirm']
PRING_BOOK_COMBINED = FEATURES + PRING_FEATURES + PRING_BOOK_FEATURES
# Chapters 16, 22-24 and 29-31: experimental intraday volume, relative-strength,
# and trend-confirmation adaptations. Market breadth is NOT inferred from two ETFs.
PRING_FULL_FEATURES = ['pring_obv_pressure', 'pring_pvt_pressure',
                       'pring_vol_trend_ratio', 'pring_effort_result',
                       'pring_range_volume', 'pring_climax_candidate',
                       'pring_trend_volume_agreement', 'pring_relative_momentum']
PRING_FULL_COMBINED = PRING_BOOK_COMBINED + PRING_FULL_FEATURES
# Nison, Japanese Candlestick Charting Techniques (2nd ed.), Chs. 3-7.
# Causal intraday research approximations; not a promise of predictive efficacy.
NISON_FEATURES = ['nison_body_fraction', 'nison_close_location', 'nison_lower_wick',
                  'nison_upper_wick', 'nison_engulfing', 'nison_hammer',
                  'nison_shooting_star', 'nison_harami', 'nison_morning_evening',
                  'nison_window', 'nison_window_hold', 'nison_reversal_confirm']
NISON_CONTROL = FEATURES
NISON_COMBINED = FEATURES + NISON_FEATURES

HORIZONS = (1,3,6)


def make_features(bars, benchmark=None):
    """Input: complete regular-session 5m bars, possibly multiple sessions."""
    if bars.empty:
        return pd.DataFrame()
    chunks=[]
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.Close.reindex(bars.index)
    else:
        aligned = None
    for _, d in bars.groupby(bars.index.date):
        d=d.copy().sort_index()
        close=d.Close.astype(float)
        e9=close.ewm(span=9,adjust=False).mean()
        e20=close.ewm(span=20,adjust=False).mean()
        e12=close.ewm(span=12,adjust=False).mean()
        e26=close.ewm(span=26,adjust=False).mean()
        macd=e12-e26
        hist=macd-macd.ewm(span=9,adjust=False).mean()
        tp=(d.High+d.Low+close)/3
        vwap=(tp*d.Volume).cumsum()/d.Volume.cumsum().replace(0,np.nan)
        rvol=d.Volume/d.Volume.shift(1).rolling(20,min_periods=10).mean().replace(0,np.nan)
        logp=np.log(close)
        z=(close-close.shift(1).rolling(20).mean())/close.shift(1).rolling(20).std().replace(0,np.nan)
        opening=d.iloc[:3]
        orhi=opening.High.max() if len(opening)==3 else np.nan
        orlo=opening.Low.min() if len(opening)==3 else np.nan
        d['ema_gap']=(e9-e20)/close*10000
        d['vwap_gap']=(close-vwap)/close*10000
        d['momentum3']=(logp-logp.shift(3))*10000
        d['momentum6']=(logp-logp.shift(6))*10000
        d['rvol']=np.log1p(rvol.clip(0,20))
        d['range_pct']=(d.High-d.Low)/close*10000
        d['structure']=((d.High>d.High.shift(1)).astype(float)-(d.Low<d.Low.shift(1)).astype(float))
        d['or_position']=np.where(np.arange(len(d))>=3,(close-(orhi+orlo)/2)/close*10000,np.nan)
        d['z20']=z.clip(-5,5)
        d['macd_hist']=hist/close*10000
        # Edwards-Magee inspired, strictly causal 5m features. All levels
        # use completed PRIOR candles; no centered pivots or future confirmation.
        prior_high=d.High.shift(1).rolling(12,min_periods=6).max()
        prior_low=d.Low.shift(1).rolling(12,min_periods=6).min()
        width=(prior_high-prior_low).replace(0,np.nan)
        true_range=pd.concat([d.High-d.Low,
                              (d.High-close.shift(1)).abs(),
                              (d.Low-close.shift(1)).abs()],axis=1).max(axis=1)
        atr=true_range.shift(1).rolling(14,min_periods=8).mean().replace(0,np.nan)
        net=(close-close.shift(6)).abs()
        path=close.diff().abs().rolling(6,min_periods=6).sum().replace(0,np.nan)
        d['trend_efficiency']=(net/path).clip(0,1)
        d['swing_position']=((close-prior_low)/width).clip(-2,3)
        d['resistance_distance']=((prior_high-close)/atr).clip(-8,8)
        d['support_distance']=((close-prior_low)/atr).clip(-8,8)
        up=(close>prior_high)&(close.shift(1)<=prior_high.shift(1))
        down=(close<prior_low)&(close.shift(1)>=prior_low.shift(1))
        d['breakout_up']=up.astype(float)
        d['breakout_down']=down.astype(float)
        # Reclaim of a broken level on the next bar: available only AFTER close.
        d['failed_up']=((close<prior_high.shift(1))&up.shift(1,fill_value=False)).astype(float)
        d['failed_down']=((close>prior_low.shift(1))&down.shift(1,fill_value=False)).astype(float)
        recent_range=(d.High.rolling(6,min_periods=6).max()-d.Low.rolling(6,min_periods=6).min())
        d['compression']=(recent_range/atr).clip(0,15)
        bar_range=(d.High-d.Low).replace(0,np.nan)
        d['wick_pressure']=((d[['Open','Close']].min(axis=1)-d.Low)-
                            (d.High-d[['Open','Close']].max(axis=1)))/bar_range
        d['volume_confirmation']=np.where(up|down,np.log1p(rvol.clip(0,20)),0.0)
        # Steve Nison candlestick research, Chapters 3-7. Every signal is
        # timestamped at its FINAL completed candle, never retroactively.
        # Intraday windows are strictly non-overlapping candle ranges; most
        # SPY/QQQ 5m bars overlap, so a zero is normal, not missing data.
        o=d.Open.astype(float); hi=d.High.astype(float); lo=d.Low.astype(float)
        rng=(hi-lo).replace(0,np.nan)
        body=(close-o).abs(); upper=hi-pd.concat([o,close],axis=1).max(axis=1)
        lower=pd.concat([o,close],axis=1).min(axis=1)-lo
        bullish=close>o; bearish=close<o
        prior_bull=o.shift(1)<close.shift(1)
        prior_bear=o.shift(1)>close.shift(1)
        prior_top=pd.concat([o.shift(1),close.shift(1)],axis=1).max(axis=1)
        prior_bottom=pd.concat([o.shift(1),close.shift(1)],axis=1).min(axis=1)
        current_top=pd.concat([o,close],axis=1).max(axis=1)
        current_bottom=pd.concat([o,close],axis=1).min(axis=1)
        # Pre-existing trend is known at prior close, not defined by the signal.
        prior_trend=close.shift(1)-close.shift(5)
        downtrend=prior_trend<0; uptrend=prior_trend>0
        engulf_up=downtrend & prior_bear & bullish & (current_bottom<=prior_bottom) & (current_top>=prior_top)
        engulf_dn=uptrend & prior_bull & bearish & (current_bottom<=prior_bottom) & (current_top>=prior_top)
        hammer=downtrend & (lower>=2*body) & (upper<=body) & (body/rng<=0.40)
        shooting=uptrend & (upper>=2*body) & (lower<=body) & (body/rng<=0.40)
        harami=(current_top<=prior_top) & (current_bottom>=prior_bottom) & (body<body.shift(1)*0.7)
        # Star approximations allow overlapping bars, as discussed for intraday
        # sessions by Nison; signal only at close of third candle.
        first_body=body.shift(2); middle_body=body.shift(1)
        morning=(bearish.shift(2).eq(True) & (middle_body<first_body*0.5) &
                 bullish & (close>(o.shift(2)+close.shift(2))/2) &
                 (close.shift(2)<close.shift(5)))
        evening=(bullish.shift(2).eq(True) & (middle_body<first_body*0.5) &
                 bearish & (close<(o.shift(2)+close.shift(2))/2) &
                 (close.shift(2)>close.shift(5)))
        window_up=(lo>hi.shift(1)); window_down=(hi<lo.shift(1))
        # Prior bar gap can only be tested for holding at this bar's close.
        window_hold=np.select([window_up.shift(1,fill_value=False)&(lo>=hi.shift(2)),
                               window_down.shift(1,fill_value=False)&(hi<=lo.shift(2))],
                              [1.0,-1.0],default=0.0)
        reversal_up=engulf_up|hammer|morning
        reversal_dn=engulf_dn|shooting|evening
        # Confirm the preceding completed candle's reversal with today's close.
        confirm_up=reversal_up.shift(1,fill_value=False)&(close>hi.shift(1))
        confirm_dn=reversal_dn.shift(1,fill_value=False)&(close<lo.shift(1))
        d['nison_body_fraction']=(body/rng).clip(0,1)
        d['nison_close_location']=((close-lo)/rng*2-1).clip(-1,1)
        d['nison_lower_wick']=(lower/rng).clip(0,1)
        d['nison_upper_wick']=(upper/rng).clip(0,1)
        d['nison_engulfing']=engulf_up.astype(float)-engulf_dn.astype(float)
        d['nison_hammer']=hammer.astype(float)
        d['nison_shooting_star']=shooting.astype(float)
        d['nison_harami']=harami.astype(float)
        d['nison_morning_evening']=morning.astype(float)-evening.astype(float)
        d['nison_window']=window_up.astype(float)-window_down.astype(float)
        d['nison_window_hold']=window_hold
        d['nison_reversal_confirm']=confirm_up.astype(float)-confirm_dn.astype(float)
        # Pring-inspired momentum confirmation / divergence research.
        # Intraday adaptations, NOT a reproduction of Pring's proprietary
        # KST parameters or a claim of calibrated predictive power.
        # Every calculation uses only the current and earlier completed bars.
        roc3 = (close / close.shift(3) - 1.0) * 10000.0
        roc6 = (close / close.shift(6) - 1.0) * 10000.0
        roc12 = (close / close.shift(12) - 1.0) * 10000.0
        d['pring_roc3'] = roc3.clip(-300, 300)
        d['pring_roc6'] = roc6.clip(-500, 500)
        d['pring_roc12'] = roc12.clip(-800, 800)
        d['pring_momentum_accel'] = (roc3 - roc3.shift(3)).clip(-500, 500)
        d['pring_momentum_confirmation'] = (
            np.sign(roc3.fillna(0)) + np.sign(roc6.fillna(0))
            + np.sign(roc12.fillna(0))) / 3.0
        # A new price extreme with an opposing ROC extreme is a divergence
        # *candidate*. Prior comparison windows exclude the current candle.
        high_prev = d.High.shift(1).rolling(12, min_periods=8).max()
        low_prev = d.Low.shift(1).rolling(12, min_periods=8).min()
        roc_prev_hi = roc6.shift(1).rolling(12, min_periods=8).max()
        roc_prev_lo = roc6.shift(1).rolling(12, min_periods=8).min()
        d['pring_bear_div'] = ((d.High > high_prev) &
                               (roc6 < roc_prev_hi)).astype(float)
        d['pring_bull_div'] = ((d.Low < low_prev) &
                               (roc6 > roc_prev_lo)).astype(float)
        # Momentum relative to its prior volatility: descriptive only.
        d['pring_roc6_z'] = ((roc6 - roc6.shift(1).rolling(20, min_periods=10).mean()) /
                               roc6.shift(1).rolling(20, min_periods=10).std().replace(0, np.nan)).clip(-5, 5)
        # Pring, Technical Analysis Explained (4th ed.), Ch. 12, Table 12-2:
        # Short-term KST uses ROC 10/15/20/30, SMA smoothing 10/10/10/15,
        # and weights 1/2/3/4. The 9-bar KST signal is an explicit research
        # convention, not a claim of a book-prescribed 5-minute signal.
        # Each component uses only the current and earlier COMPLETED bars.
        kst_parts = []
        for period, smoothing, weight in [(10,10,1),(15,10,2),(20,10,3),(30,15,4)]:
            component = (100.0 * (close / close.shift(period) - 1.0)).rolling(
                smoothing, min_periods=smoothing).mean()
            kst_parts.append(weight * component)
        kst = sum(kst_parts)
        kst_signal = kst.rolling(9, min_periods=9).mean()
        d['pring_kst'] = kst.clip(-100,100)
        d['pring_kst_signal_gap'] = (kst - kst_signal).clip(-30,30)
        d['pring_kst_slope'] = (kst - kst.shift(3)).clip(-30,30)
        # Pring Ch. 12, Directional Movement System. Wilder-style DI and ADX,
        # calculated causally within each regular session; warmup is mandatory.
        up_move = d.High.diff()
        down_move = -d.Low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = pd.concat([d.High-d.Low, (d.High-close.shift(1)).abs(),
                        (d.Low-close.shift(1)).abs()], axis=1).max(axis=1)
        # Wilder smoothing with min_periods ensures no early unstable values.
        smooth_tr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().replace(0,np.nan)
        pdi = 100 * plus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/smooth_tr
        mdi = 100 * minus_dm.ewm(alpha=1/14,adjust=False,min_periods=14).mean()/smooth_tr
        dx = 100 * (pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
        d['pring_adx14'] = dx.ewm(alpha=1/14,adjust=False,min_periods=14).mean().clip(0,100)
        d['pring_di_spread'] = (pdi-mdi).clip(-100,100)
        # Pring Ch. 10: momentum warnings should be confirmed by price.
        # Confirm momentum direction with a completed close beyond PRIOR 6-bar range.
        prior6hi = d.High.shift(1).rolling(6,min_periods=6).max()
        prior6lo = d.Low.shift(1).rolling(6,min_periods=6).min()
        d['pring_roc_price_confirm'] = np.select(
            [(roc6 > 0) & (close > prior6hi), (roc6 < 0) & (close < prior6lo)],
            [1.0,-1.0], default=0.0)
        # Relative strength feature: historical QQQ/SPY or SPY/QQQ ratio,
        # causal, synchronized completed 5m bars. Do not forward-fill missing bars.
        if aligned is not None:
            bench=aligned.reindex(d.index)
            rs=close/bench
            d['ws_rs_15m']=np.log(rs/rs.shift(3))*10000
            d['ws_rs_60m']=np.log(rs/rs.shift(12))*10000
        else:
            d['ws_rs_15m']=np.nan
            d['ws_rs_60m']=np.nan
        # Pring Chs. 22-23: volume as trend confirmation, not an independent
        # directional oracle. These are CAUSAL intraday adaptations, not
        # book-prescribed 5-minute trading rules.
        vol = d.Volume.astype(float)
        vol_ref = vol.shift(1).rolling(20, min_periods=12).mean().replace(0,np.nan)
        price_move = close.diff()
        signed_vol = np.sign(price_move.fillna(0)) * vol
        obv = signed_vol.cumsum()
        pvt = (price_move / close.shift(1).replace(0,np.nan) * vol).fillna(0).cumsum()
        d['pring_obv_pressure'] = ((obv-obv.shift(6)) / (vol_ref*6)).clip(-5,5)
        d['pring_pvt_pressure'] = ((pvt-pvt.shift(6)) / (vol_ref*6)).clip(-1,1)
        d['pring_vol_trend_ratio'] = (vol.rolling(5,min_periods=5).mean() /
            vol.shift(5).rolling(15,min_periods=12).mean().replace(0,np.nan)).clip(0,10)
        # Relative volume versus prior completed bars, with normalized price progress.
        d['pring_effort_result'] = ((close-close.shift(3)) / atr /
            (vol.rolling(3,min_periods=3).mean()/vol_ref).replace(0,np.nan)).clip(-10,10)
        d['pring_range_volume'] = (true_range / atr * (vol/vol_ref)).clip(0,30)
        d['pring_climax_candidate'] = ((vol/vol_ref > 2.0) &
            (true_range/atr > 1.8)).astype(float)
        trend6 = np.sign(close-close.shift(6))
        d['pring_trend_volume_agreement'] = (trend6 *
            (obv-obv.shift(6))/(vol_ref*6)).clip(-5,5)
        # Pring Ch. 16: relative strength vs a synchronized benchmark.
        # A missing benchmark remains NaN, never silently forward-filled.
        if aligned is not None:
            bclose = aligned.reindex(d.index)
            ratio = close/bclose
            d['pring_relative_momentum'] = (np.log(ratio/ratio.shift(6))*10000).clip(-500,500)
        else:
            d['pring_relative_momentum'] = np.nan
        # Weinstein-inspired INTRADAY proxies, not original 30-week stages.
        # Rolling resistance and volume references exclude the signal candle.
        ws_ma = close.ewm(span=30, adjust=False).mean()
        ws_prior_hi = d.High.shift(1).rolling(18, min_periods=12).max()
        ws_prior_lo = d.Low.shift(1).rolling(18, min_periods=12).min()
        ws_atr = true_range.shift(1).rolling(14, min_periods=8).mean().replace(0, np.nan)
        ws_vol = d.Volume.shift(1).rolling(20, min_periods=10).mean().replace(0, np.nan)
        d['ws_ma_slope'] = ((ws_ma - ws_ma.shift(6)) / ws_atr).clip(-10, 10)
        d['ws_base_width'] = ((ws_prior_hi - ws_prior_lo) / ws_atr).clip(0, 30)
        d['ws_breakout_strength'] = ((close - ws_prior_hi) / ws_atr).clip(-10, 10)
        d['ws_pullback_quality'] = ((close - ws_ma) / ws_atr).clip(-10, 10)
        d['ws_overhead_atr'] = ((ws_prior_hi - close) / ws_atr).clip(-10, 10)
        d['ws_volume_on_break'] = np.where((close > ws_prior_hi) | (close < ws_prior_lo),
                                           np.log1p((d.Volume / ws_vol).clip(0, 20)), 0.0)
        for h in HORIZONS:
            d[f'y{h}']=np.log(close.shift(-h)/close)*10000
            # Never label across a session boundary.
        chunks.append(d)
    return pd.concat(chunks).sort_index()


def _fit(X,y,lam=2.0,steps=30):
    """Regularized logistic regression via Newton steps, numpy only."""
    w=np.zeros(X.shape[1]); eye=np.eye(len(w)); eye[0,0]=0
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(X@w,-30,30)))
        h=X.T@((p*(1-p))[:,None]*X)+lam*eye+1e-7*np.eye(len(w))
        g=X.T@(p-y)-lam*(-eye@w)
        step=np.linalg.solve(h,g)
        w-=step
        if np.max(np.abs(step))<1e-6:break
    return w


def _standardize(train,test):
    mu=train.mean(axis=0); sd=train.std(axis=0)
    sd=np.where(sd<1e-8,1,sd)
    a=np.column_stack([np.ones(len(train)),np.clip((train-mu)/sd,-8,8)])
    b=np.column_stack([np.ones(len(test)),np.clip((test-mu)/sd,-8,8)])
    return a,b


def evaluate_forward(bars, cost_bps=4.0, min_train=150, min_test=40, features=None, benchmark=None):
    """Chronological 70/30 held-out evaluation, with embargo per horizon.
    Latest inference trains only on rows with realized outcomes available.
    Results are descriptive and NOT calibrated probabilities.
    """
    f=make_features(bars, benchmark=benchmark)
    features = FEATURES if features is None else features
    if f.empty:return {}
    out={}
    for h in HORIZONS:
        target=f[f'y{h}']
        valid=f[features].notna().all(axis=1)&target.notna()
        history=f.loc[valid].copy()
        latest=f.iloc[-1]
        if len(history)<min_train+min_test or not latest[features].notna().all():
            out[h]={'status':'insufficient','n':len(history)}
            continue
        # Separate by session dates so test sessions never appear in training.
        days=np.array(sorted(set(history.index.date)))
        if len(days)<8:
            out[h]={'status':'insufficient','n':len(history)};continue
        split_day=days[max(1,int(len(days)*.7))]
        train=history[history.index.date<split_day]
        test=history[history.index.date>=split_day]
        # Guard the train/test boundary and overlapping forward labels.
        train=train.iloc[:-h] if len(train)>h else train.iloc[:0]
        if len(train)<min_train or len(test)<min_test:
            out[h]={'status':'insufficient','n':len(history),'train':len(train),'test':len(test)};continue
        xtr,xt=_standardize(train[features].to_numpy(float),test[features].to_numpy(float))
        w=_fit(xtr,(train[f'y{h}']>0).astype(float).to_numpy())
        prob=1/(1+np.exp(-np.clip(xt@w,-30,30)))
        actual=test[f'y{h}'].to_numpy(float)
        pred=np.where(prob>=.5,1,-1)
        # Sparse, nonoverlapping evaluation observations per day, to avoid inflated sample counts.
        take=[]
        for _,grp in test.groupby(test.index.date):
            positions=test.index.get_indexer(grp.index)
            take.extend(positions[::h].tolist())
        ix=np.array(take,dtype=int)
        accuracy=float(np.mean((pred[ix]*actual[ix])>0))
        net=pred[ix]*actual[ix]-cost_bps
        # Fresh fit on all labeled data for current inference, excluding unknown future outcomes.
        full=history[features].to_numpy(float)
        a,b=_standardize(full,latest[features].to_numpy(float).reshape(1,-1))
        wf=_fit(a,(history[f'y{h}']>0).astype(float).to_numpy())
        current=float(1/(1+np.exp(-np.clip((b@wf)[0],-30,30))))
        oos_p=prob[ix]
        brier=float(np.mean((oos_p-(actual[ix]>0).astype(float))**2))
        base_rate=float(np.mean(train[f'y{h}']>0))
        base_brier=float(np.mean((base_rate-(actual[ix]>0).astype(float))**2))
        out[h]={'status':'ok','p_up':current,'p_down':1-current,'n_train':len(train),
                'n_test':len(ix),'accuracy':accuracy,'mean_net_bps':float(np.mean(net)),
                'brier':brier,'baseline_brier':base_brier,
                'up_base_rate':base_rate,'last_bar':str(f.index[-1]),'last_close':float(f.iloc[-1]['Close']),
                'test_start':str(test.index[0]),'test_end':str(test.index[-1]),
                'cost_bps':cost_bps,
                'oos_prob_up':oos_p.tolist(),
                'oos_forward_bps':actual[ix].tolist()}
    return out


@st.cache_data(show_spinner=False)
def _forward_history(sym: str, bucket: int, days: int):
    raw=fetch_bars_1m(sym,bucket,days=days)
    if raw.empty:return raw
    five=resample_5min(raw)
    now=pd.Timestamp.now(tz=ET)
    # Drop incomplete last bar of current session.
    if len(five) and five.index[-1].date()==now.date():
        five=five[five.index+pd.Timedelta(minutes=5)<=now]
    return five

def _forecast_label(r: dict, price_bias: str = 'Mixed') -> tuple[str, str]:
    """Separate model hypothesis from evidence-backed trade readiness."""
    if r.get('status') != 'ok':
        return 'NEUTRAL / NO EDGE', 'Insufficient labeled history.'
    p = float(r['p_up'])
    passed = (r['mean_net_bps'] > 0 and r['accuracy'] >= .53
              and r['brier'] < r['baseline_brier'] and r['n_test'] >= 40)
    if not passed:
        return 'NEUTRAL / NO EDGE', 'Historical holdout has not demonstrated a reliable net advantage.'
    # Probability separation is a provisional abstention rule, not calibration.
    if abs(p - .5) < .10:
        return 'NEUTRAL / NO EDGE', 'Model probability is too close to 50/50.'
    direction = 'BULLISH' if p > .5 else 'BEARISH'
    if price_bias != 'Mixed' and direction.title() != price_bias:
        return 'NEUTRAL / NO EDGE', f'Model hypothesis conflicts with {price_bias.lower()} price structure.'
    return direction + ' (RESEARCH ONLY)', 'Provisional research candidate; not a validated live signal.'


def _record_forecasts(sym: str, results: dict, five: pd.DataFrame):
    """Session-local forecast ledger; only score after target bar has closed.

    Records persist across Streamlit fragment reruns, not server restarts.
    No predictions are reconstructed retrospectively from realized outcomes.
    """
    if five.empty:
        return []
    key = f'forecast_ledger_{sym}'
    ledger = st.session_state.setdefault(key, [])
    latest_ts = five.index[-1]
    close = float(five.Close.iloc[-1])
    # A 5-minute bar is labeled by its start timestamp; predictions are made
    # after its close. Score at the close of the h-th subsequent bar.
    for item in ledger:
        if item.get('actual') is not None:
            continue
        target = pd.Timestamp(item['bar_start']) + pd.Timedelta(minutes=5 * item['h'])
        # Never score across a session boundary.
        if target.date() != pd.Timestamp(item['bar_start']).date() or (latest_ts.date() > pd.Timestamp(item['bar_start']).date() and target not in five.index):
            item['actual'] = 'EXPIRED'; continue
        if target in five.index and latest_ts >= target:
            end_close = float(five.loc[target, 'Close'])
            change_bps = (end_close / item['start_price'] - 1) * 10000
            item['actual'] = 'UP' if change_bps > 0 else 'DOWN' if change_bps < 0 else 'FLAT'
            item['move_bps'] = round(change_bps, 2)
            item['correct'] = (item['actual'] == item['hypothesis'])
    for h in HORIZONS:
        r = results.get(h, {})
        if r.get('status') != 'ok': continue
        # Do not create a forecast from a historical bar when opening the app
        # well after its close; only track predictions observed in the session.
        now = pd.Timestamp.now(tz=ET)
        age = (now - (latest_ts + pd.Timedelta(minutes=5))).total_seconds() / 60
        if not (in_market_hours() and -1 <= age <= 10): continue
        if any(item['bar_start'] == latest_ts.isoformat() and item['h'] == h for item in ledger):
            continue
        p = float(r['p_up'])
        ledger.append({'bar_start': latest_ts.isoformat(), 'h': h,
                       'start_price': close, 'p_up': round(p, 4),
                       'hypothesis': 'UP' if p >= .5 else 'DOWN',
                       'display': _forecast_label(r)[0], 'actual': None})
    st.session_state[key] = ledger[-300:]
    return ledger


def probability_reliability(r: dict, min_samples: int = 30) -> pd.DataFrame:
    """Reliability of held-out DIRECTION confidence, not calibrated probability.

    Bins are disjoint [50,60), ..., [90,100]. All statistics use the same
    non-overlapping out-of-sample observations as the horizon evaluation.
    Wilson 95% interval measures sampling uncertainty, not market uncertainty.
    """
    columns = ['Confidence band', 'n', 'Mean predicted confidence',
               'Realized hit rate', '95% CI low', '95% CI high',
               'Mean net bps', 'Brier (UP)', 'Assessment']
    if r.get('status') != 'ok' or 'oos_prob_up' not in r:
        return pd.DataFrame(columns=columns)
    p = np.asarray(r['oos_prob_up'], dtype=float)
    moves = np.asarray(r['oos_forward_bps'], dtype=float)
    good = np.isfinite(p) & np.isfinite(moves)
    p, moves = p[good], moves[good]
    confidence = np.maximum(p, 1-p)
    predicted_up = p >= 0.5
    correct = np.where(predicted_up, moves > 0, moves < 0)
    net = np.where(predicted_up, moves, -moves) - float(r['cost_bps'])
    rows = []
    for lower in (0.5, 0.6, 0.7, 0.8, 0.9):
        upper = lower + 0.1
        mask = (confidence >= lower - 1e-10) & (confidence < upper - 1e-10 if upper < 1 else confidence <= 1)
        n = int(mask.sum())
        band = f'{int(lower*100)}–{int(round(upper*100))}%'
        if n == 0:
            rows.append(dict(zip(columns, [band, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'No observations'])))
            continue
        hit = float(correct[mask].mean())
        z = 1.96
        denom = 1 + z*z/n
        center = (hit + z*z/(2*n))/denom
        margin = z*np.sqrt(hit*(1-hit)/n + z*z/(4*n*n))/denom
        brier = float(np.mean((p[mask] - (moves[mask] > 0).astype(float))**2))
        rows.append(dict(zip(columns, [band, n, float(confidence[mask].mean()),
            hit, max(0., center-margin), min(1., center+margin),
            float(net[mask].mean()), brier,
            'Exploratory only' if n >= min_samples else 'Insufficient sample'])) )
    return pd.DataFrame(rows, columns=columns)


def render_probability_reliability(sym: str, results: dict):
    with st.expander(f'{sym} — Probability Reliability Check', expanded=True):
        st.caption('Are confident forecasts actually correct that often? '
                   'These are raw, uncalibrated predictions tested on the chronological holdout. '
                   'A strong raw percentage is NOT a validated probability or a trade signal.')
        horizon = st.selectbox('Forecast horizon', [5, 15, 30], index=2,
                               key=f'reliability_horizon_{sym}')
        r = results.get(horizon // 5, {})
        if r.get('status') != 'ok':
            st.info('Not enough out-of-sample data to assess reliability.')
            return
        table = probability_reliability(r)
        p_up = float(r['p_up'])
        confidence = max(p_up, 1-p_up)
        band_index = min(4, max(0, int((confidence-0.5)*10)))
        selected = table.iloc[band_index]
        direction = 'UP' if p_up >= 0.5 else 'DOWN'
        st.markdown(f'**Current raw hypothesis: {direction} {confidence:.1%}** · '
                    f'Comparable holdout confidence band: **{selected["Confidence band"]}**')
        if int(selected['n']) < 30:
            st.warning(f'Only {int(selected["n"])} non-overlapping holdout examples in this band. '
                       'Insufficient evidence to assess this confidence level. NO VALIDATED EDGE.')
        else:
            st.markdown(f'**Historical hit rate in this band: {selected["Realized hit rate"]:.1%}** '
                        f'(95% Wilson interval {selected["95% CI low"]:.1%}–{selected["95% CI high"]:.1%}; '
                        f'n={int(selected["n"])}). Mean net return '
                        f'{selected["Mean net bps"]:+.2f} bps per hypothetical directional trade.')
            st.caption('This is an exploratory conditional result from one holdout, not a separately validated '
                       'or calibrated live probability. The current forecast was fitted on more data than the '
                       'holdout forecasts, so the band comparison is approximate.')
        display = table.copy()
        for c in ['Mean predicted confidence', 'Realized hit rate', '95% CI low', '95% CI high']:
            display[c] = display[c].map(lambda v: f'{v:.1%}' if pd.notna(v) else '—')
        for c in ['Mean net bps', 'Brier (UP)']:
            display[c] = display[c].map(lambda v: f'{v:+.2f}' if pd.notna(v) and c == 'Mean net bps' else (f'{v:.3f}' if pd.notna(v) else '—'))
        st.dataframe(display, hide_index=True, use_container_width=True)
        chart_data = table[table['n'] >= 30]
        if not chart_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[.5, 1], y=[.5, 1], mode='lines',
                                     name='Perfect calibration', line=dict(color='#9ca3af', dash='dash')))
            fig.add_trace(go.Scatter(x=chart_data['Mean predicted confidence'],
                                     y=chart_data['Realized hit rate'], mode='markers+lines',
                                     name='Holdout observed',
                                     error_y=dict(type='data', symmetric=False,
                                                  array=(chart_data['95% CI high']-chart_data['Realized hit rate']).to_list(),
                                                  arrayminus=(chart_data['Realized hit rate']-chart_data['95% CI low']).to_list()),
                                     marker=dict(size=9)))
            fig.update_layout(height=300, margin=dict(l=35,r=25,t=20,b=35),
                              xaxis=dict(title='Mean raw directional confidence', tickformat='.0%',range=[.48,1.02]),
                              yaxis=dict(title='Actual directional hit rate',tickformat='.0%',range=[0,1.02]),
                              legend=dict(orientation='h',y=1.15))
            st.plotly_chart(fig, use_container_width=True, key=f'reliability_chart_{sym}')
        st.caption('Bands use non-overlapping holdout observations per session, selected before examining '
                   'outcomes. Wilson intervals assume independent trials and may understate uncertainty '
                   'because trading outcomes cluster by day. Brier scores use the UP probability, '
                   'not the directional confidence. Net bps assumes configured round-trip costs. '
                   'No probability calibration or walk-forward retraining has been performed.')


def render_forward(sym: str, five: pd.DataFrame | None = None, price_bias: str = 'Mixed'):
    st.subheader(f'{sym} | Forward Direction Engine')
    st.caption('Research forecasts, not trade instructions. Directional probabilities are uncalibrated.')
    with st.expander('Forecast research settings'):
        days = st.slider('Historical calendar days', 20, 90, 45, 5, key=f'forecast_days_{sym}')
        cost = st.number_input('Assumed round-trip trading costs (basis points)', 0.0, 30.0, 4.0, 0.5, key=f'forecast_cost_{sym}')
    try:
        history = _forward_history(sym, _bucket(live, int(refresh_secs)), days)
        results = evaluate_forward(history, cost_bps=cost, features=FEATURES)
        baseline = evaluate_forward(history, cost_bps=cost, features=BASE_FEATURES)
        weinstein = evaluate_forward(history, cost_bps=cost, features=WEINSTEIN_COMBINED)
        other_sym = 'SPY' if sym == 'QQQ' else 'QQQ'
        other_history = _forward_history(other_sym, _bucket(live, int(refresh_secs)), days)
        rs_model = evaluate_forward(history, cost_bps=cost,
                                    features=WEINSTEIN_ORIGINAL_TEST,
                                    benchmark=other_history)
    except Exception as exc:
        st.warning(f'Forecast research unavailable: {exc}')
        return
    if not results:
        st.info('Historical bars unavailable; no forecast can be calculated.')
        return
    if history.empty:
        return
    last_end = history.index[-1] + pd.Timedelta(minutes=5)
    st.caption(f'Forecast source: last completed 5-minute candle ending {last_end.strftime("%Y-%m-%d %H:%M ET")}. '
               f'Current price structure: {price_bias}.')
    now = pd.Timestamp.now(tz=ET)
    if in_market_hours() and (now - last_end).total_seconds() > 600:
        st.warning('Forecast data are stale (>10 minutes); do not interpret as a current prediction.')
    cols = st.columns(3)
    for col, h in zip(cols, HORIZONS):
        r = results.get(h, {})
        with col:
            st.markdown(f'**Next {h*5} minutes**')
            if r.get('status') != 'ok':
                st.markdown('**NEUTRAL / NO EDGE**')
                st.caption(f'Insufficient labeled history (n={r.get("n",0)}).')
                continue
            label, explanation = _forecast_label(r, price_bias)
            st.markdown(f'**{label}**')
            st.caption(f'Raw hypothesis: {"UP" if r["p_up"] >= .5 else "DOWN"} · '
                       f'UP {r["p_up"]:.1%} / DOWN {r["p_down"]:.1%}')
            st.caption(explanation)
            st.caption(f'Holdout n={r["n_test"]}; accuracy {r["accuracy"]:.1%}; '
                       f'net {r["mean_net_bps"]:+.1f} bps/trade; '
                       f'Brier {r["brier"]:.3f} vs training-base {r["baseline_brier"]:.3f}.')
    with st.expander('Weinstein-inspired feature test — baseline vs expanded model', expanded=False):
        st.caption('The original V6 forecast above is unchanged. The experimental Weinstein-expanded model is evaluated separately on the same history. The book uses weekly stages; these are explicitly intraday proxies.')
        rows = []
        for h in HORIZONS:
            old, new = results.get(h, {}), weinstein.get(h, {})
            rows.append({'Horizon': f'{h*5}m',
                         'V6 accuracy': f'{old["accuracy"]:.1%}' if old.get('status') == 'ok' else 'N/A',
                         'Weinstein accuracy': f'{new["accuracy"]:.1%}' if new.get('status') == 'ok' else 'N/A',
                         'V6 net bps': round(old['mean_net_bps'], 2) if old.get('status') == 'ok' else None,
                         'Weinstein net bps': round(new['mean_net_bps'], 2) if new.get('status') == 'ok' else None,
                         'V6 Brier': round(old['brier'], 3) if old.get('status') == 'ok' else None,
                         'Weinstein Brier': round(new['brier'], 3) if new.get('status') == 'ok' else None,
                         'V6 n': old.get('n_test'), 'Weinstein n': new.get('n_test')})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption('Different feature warmups can produce different samples; this comparison is exploratory, not a matched-sample causal test. Do not trade based on the expanded model until independently validated.')
    with st.expander('Original Weinstein: relative-strength feature ablation', expanded=False):
        st.caption('New experimental model adds causal 15m/60m ETF-relative-strength features to the Weinstein intraday model. It does NOT substitute intraday bars for the original 30-week stage. Forecasts above remain unchanged.')
        rows=[]
        for h in HORIZONS:
            old,new=weinstein.get(h,{}),rs_model.get(h,{})
            rows.append({'Horizon':f'{5*h}m',
                         'Without RS accuracy':f'{old["accuracy"]:.1%}' if old.get('status')=='ok' else 'N/A',
                         'With RS accuracy':f'{new["accuracy"]:.1%}' if new.get('status')=='ok' else 'N/A',
                         'Without RS net bps':round(old['mean_net_bps'],2) if old.get('status')=='ok' else None,
                         'With RS net bps':round(new['mean_net_bps'],2) if new.get('status')=='ok' else None,
                         'Without RS Brier':round(old['brier'],3) if old.get('status')=='ok' else None,
                         'With RS Brier':round(new['brier'],3) if new.get('status')=='ok' else None,
                         'With RS n':new.get('n_test')})
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        st.caption('Exploratory single chronological holdout, not rolling walk-forward; different warmups may change the evaluated sample. Positive numbers do not establish a tradable edge. Raw probabilities are uncalibrated.')
    with st.expander('Pring book-based momentum: KST, DI/ADX and price confirmation', expanded=False):
        st.caption('Source: Technical Analysis Explained, Fourth Edition, Chapters 10-12 (uploaded scan, PDF pp. 98-136). KST uses the short-term row of Table 12-2; five-minute application is experimental. The existing forecast above is unchanged.')
        try:
            pring = evaluate_forward(history, cost_bps=cost, features=PRING_COMBINED)
            book = evaluate_forward(history, cost_bps=cost, features=PRING_BOOK_COMBINED)
            table=[]
            for h in HORIZONS:
                r0,r1,r2=results.get(h,{}),pring.get(h,{}),book.get(h,{})
                def cell(r,key,fmt=None):
                    if r.get('status')!='ok': return 'N/A'
                    val=r.get(key)
                    return fmt.format(val) if fmt else val
                table.append({'Horizon':f'{h*5}m',
                              'Baseline accuracy':cell(r0,'accuracy','{:.1%}'),
                              'Earlier Pring accuracy':cell(r1,'accuracy','{:.1%}'),
                              'Book-based accuracy':cell(r2,'accuracy','{:.1%}'),
                              'Baseline net bps':cell(r0,'mean_net_bps','{:+.2f}'),
                              'Book-based net bps':cell(r2,'mean_net_bps','{:+.2f}'),
                              'Baseline Brier':cell(r0,'brier','{:.3f}'),
                              'Book-based Brier':cell(r2,'brier','{:.3f}'),
                              'Book holdout n':r2.get('n_test','N/A')})
            st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
            latest=make_features(history)
            if not latest.empty:
                row=latest.iloc[-1]
                def fmt(name,unit=''):
                    v=row.get(name,np.nan)
                    return f'{v:+.2f}{unit}' if pd.notna(v) else 'N/A (warmup)'
                a,b,c=st.columns(3)
                a.metric('KST (5m adaptation)',fmt('pring_kst'))
                b.metric('KST vs signal',fmt('pring_kst_signal_gap'))
                c.metric('ADX (14)',fmt('pring_adx14'))
                a,b,c=st.columns(3)
                a.metric('+DI minus -DI',fmt('pring_di_spread'))
                b.metric('KST change (3 bars)',fmt('pring_kst_slope'))
                direction=row.get('pring_roc_price_confirm',0)
                c.metric('Price confirmation', 'DOWN' if direction == -1 else 'UP' if direction == 1 else 'NONE')
                st.caption('ADX measures trend strength, not direction. A KST crossover or divergence is a warning, not a verified entry. Price confirmation uses the prior six completed bars.')
            st.caption('Single chronological holdout only; samples may differ due to indicator warmup. Do not interpret raw probabilities as calibrated or promote this research model to trade readiness without matched-sample, walk-forward validation and costs.')
        except Exception as exc:
            st.info(f'Pring book comparison unavailable: {exc}')
    with st.expander('Pring full-book research — volume, relative strength and trend quality', expanded=False):
        st.caption('Source: Technical Analysis Explained, 4th ed., Chs. 16, 22-24, 29-31. '
                   'These are causal 5-minute adaptations. No advance/decline breadth is calculated: '
                   'that requires constituent-level market data. The main forecasts above are unchanged.')
        try:
            # Both comparisons use identical feature warmup and the same benchmark;
            # this is an ablation of the new volume/RS features, not a calibrated forecast.
            control_features = PRING_FULL_COMBINED[:-len(PRING_FULL_FEATURES)]
            control = evaluate_forward(history, cost_bps=cost,
                features=control_features, benchmark=other_history)
            expanded = evaluate_forward(history, cost_bps=cost,
                features=PRING_FULL_COMBINED, benchmark=other_history)
            rows=[]
            for h in HORIZONS:
                a,b=control.get(h,{}),expanded.get(h,{})
                def cell(r,k,fmt):
                    return fmt.format(r[k]) if r.get('status')=='ok' else 'N/A'
                rows.append({'Horizon':f'{5*h}m',
                    'Control accuracy':cell(a,'accuracy','{:.1%}'),
                    'Full-book accuracy':cell(b,'accuracy','{:.1%}'),
                    'Control net bps':cell(a,'mean_net_bps','{:+.2f}'),
                    'Full-book net bps':cell(b,'mean_net_bps','{:+.2f}'),
                    'Control Brier':cell(a,'brier','{:.3f}'),
                    'Full-book Brier':cell(b,'brier','{:.3f}'),
                    'Control n':a.get('n_test','N/A'),
                    'Expanded n':b.get('n_test','N/A')})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            feat=make_features(history,benchmark=other_history)
            if not feat.empty:
                z=feat.iloc[-1]
                cols=st.columns(3)
                for col,name,title in zip(cols,
                    ['pring_obv_pressure','pring_vol_trend_ratio','pring_relative_momentum'],
                    ['OBV pressure (6 bars)','Volume trend ratio','Relative momentum vs other ETF']):
                    v=z.get(name,np.nan)
                    col.metric(title, f'{v:+.2f}' if pd.notna(v) else 'N/A')
                cols=st.columns(3)
                for col,name,title in zip(cols,
                    ['pring_trend_volume_agreement','pring_effort_result','pring_climax_candidate'],
                    ['Trend-volume agreement','Price progress / volume','Climax candidate']):
                    v=z.get(name,np.nan)
                    col.metric(title, f'{v:+.2f}' if pd.notna(v) else 'N/A')
            st.caption('A volume climax is a warning, not a reversal prediction. '
                       'Single chronological holdout, potentially different warmups and small samples; '
                       'do not interpret raw probabilities as calibrated or infer a tradable edge.')
        except Exception as exc:
            st.info(f'Full-book Pring comparison unavailable: {exc}')
    with st.expander('Steve Nison — candlestick context and predictive feature test', expanded=False):
        st.caption('Source: Japanese Candlestick Charting Techniques, 2nd ed., Chs. 3–7. '
                   'Experimental completed-5m adaptations. The primary forecasts above remain unchanged.')
        try:
            # Matched comparison: use the SAME feature-complete rows for both
            # arms by including Nison columns in both sets; the control columns
            # are neutralized in the fit, preserving identical warmup/holdout.
            nison_frame=make_features(history)
            latest_n=nison_frame.iloc[-1]
            names={'nison_engulfing':'Engulfing', 'nison_hammer':'Hammer',
                   'nison_shooting_star':'Shooting star',
                   'nison_morning_evening':'Morning/evening star',
                   'nison_harami':'Harami', 'nison_window':'Rising/falling window',
                   'nison_window_hold':'Window held',
                   'nison_reversal_confirm':'Next-bar price confirmation'}
            observations=[]
            for k,label in names.items():
                val=latest_n.get(k,np.nan)
                if pd.notna(val) and abs(float(val))>0:
                    observations.append(f'{label}: {"bullish" if val>0 else "bearish"}')
            st.write('**Last completed candle:** '+('; '.join(observations) if observations else
                     'No selected candlestick event. This does not imply a sideways market.'))
            st.caption('A pattern is an observation, not a calibrated probability or automatic trade entry. '
                       'Reversal warnings may occur within a strong continuing trend.')
            control=evaluate_forward(history,cost_bps=cost,features=NISON_CONTROL)
            expanded=evaluate_forward(history,cost_bps=cost,features=NISON_COMBINED)
            rows=[]
            for h in HORIZONS:
                a,b=control.get(h,{}),expanded.get(h,{})
                def cell(r,k,fmt):
                    return fmt.format(r[k]) if r.get('status')=='ok' else 'Insufficient'
                rows.append({'Horizon':f'{5*h}m',
                    'Original accuracy':cell(a,'accuracy','{:.1%}'),
                    'With candles accuracy':cell(b,'accuracy','{:.1%}'),
                    'Original net bps':cell(a,'mean_net_bps','{:+.2f}'),
                    'With candles net bps':cell(b,'mean_net_bps','{:+.2f}'),
                    'Original Brier':cell(a,'brier','{:.3f}'),
                    'With candles Brier':cell(b,'brier','{:.3f}'),
                    'Original n':a.get('n_test','—'), 'With candles n':b.get('n_test','—')})
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            st.caption('Chronological holdout, not repeated walk-forward validation. '
                       'Different warmup/sample sizes may make comparisons unfair; '
                       'no claim of a tradable edge. Avoid acting on raw confidence.')
        except Exception as exc:
            st.info(f'Nison research unavailable: {exc}')
    render_probability_reliability(sym, results)
    ledger = _record_forecasts(sym, results, history)
    with st.expander(f'{sym} — timestamped forecast outcomes', expanded=False):
        st.caption('This ledger tracks forecasts made while the app is open. '
                   'It is session-local, resets on a new browser/server session, and is not a historical backtest.')
        if ledger:
            completed = [i for i in ledger if i.get('correct') is not None]
            if completed:
                st.write(f'Observed forecasts: {len(completed)} · raw directional accuracy '
                         f'{sum(i["correct"] for i in completed)/len(completed):.1%} '
                         '(small and overlapping samples; not validated performance).')
            rows = [{ 'Bar start (ET)': pd.Timestamp(i['bar_start']).strftime('%m-%d %H:%M'),
                      'Horizon': f'{i["h"]*5}m', 'Starting price': i['start_price'],
                      'Raw hypothesis': i['hypothesis'], 'P(up)': i['p_up'],
                      'Displayed decision': i['display'],
                      'Outcome': i['actual'] or 'PENDING',
                      'Actual move (bps)': i.get('move_bps') } for i in ledger[-60:][::-1]]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.info('No live forecasts recorded yet. Forecasts will be logged after new completed 5-minute candles.')
    with st.expander('Edwards–Magee feature comparison and limitations', expanded=False):
        comparison = []
        for h in HORIZONS:
            a, b = baseline.get(h, {}), results.get(h, {})
            comparison.append({'Horizon': f'{h*5}m',
                'Baseline accuracy': f'{a["accuracy"]:.1%}' if a.get('status') == 'ok' else 'insufficient',
                'Magee accuracy': f'{b["accuracy"]:.1%}' if b.get('status') == 'ok' else 'insufficient',
                'Baseline net bps': round(a['mean_net_bps'], 2) if a.get('status') == 'ok' else None,
                'Magee net bps': round(b['mean_net_bps'], 2) if b.get('status') == 'ok' else None,
                'Baseline Brier': round(a['brier'], 3) if a.get('status') == 'ok' else None,
                'Magee Brier': round(b['brier'], 3) if b.get('status') == 'ok' else None,
                'Baseline n': a.get('n_test') if a.get('status') == 'ok' else None,
                'Magee n': b.get('n_test') if b.get('status') == 'ok' else None})
        st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
        st.caption('Chronological session-level 70/30 holdout, non-overlapping evaluation observations. '
                   'The baseline Brier reference uses the training prevalence (not the test prevalence). '
                   'This is not repeated walk-forward retraining or probability calibration. '
                   'Costs are simplified and forecasts are not validated trading signals.')


def render_aziz(sym: str, day1: pd.DataFrame, prev: pd.DataFrame | None):
    st.subheader(f'{sym} | Price action & strategy research')
    x=price_action_features(completed_5m(day1),prev)
    if len(x)<4:
        st.info('Waiting for at least four completed 5-minute bars.')
        return
    last=x.iloc[-1]
    trend=('HH/HL' if last.HHHL else 'LH/LL' if last.LLLH else 'mixed')
    volume=f'{last.RVOL20:.2f}x' if pd.notna(last.RVOL20) else 'warming up'
    c1,c2,c3,c4=st.columns(4)
    c1.metric('5m price structure',trend)
    c2.metric('RVOL vs prior 20 bars',volume)
    c3.metric('Price vs VWAP', 'above' if last.Close>last.VWAP else 'below')
    c4.metric('EMA9 / EMA20','bull aligned' if last.EMA9>last.EMA20 else 'bear aligned')
    render_decision(sym, x)
    if show_forecast:
        render_forward(sym)
    show_marks = st.checkbox('Show recent observations on chart', value=True, key=f'aziz_marks_{sym}')
    mark_count = st.slider('Recent observation markers', 1, 15, 6, key=f'aziz_count_{sym}') if show_marks else 6
    st.plotly_chart(aziz_chart(sym,x,show_events=show_marks,event_count=mark_count),use_container_width=True,key=f'aziz_{sym}')
    st.caption('Green ▲ = upward observation; red ▼ = downward observation. Hover over markers for event details. Markers are not trade recommendations.')
    patterns=[name for col,name in [('Doji','doji / indecision'),('BullEngulf','bullish engulfing'),
              ('BearEngulf','bearish engulfing'),('BullReject','lower-wick rejection'),
              ('BearReject','upper-wick rejection')] if bool(last[col])]
    st.write('**Last completed candle:** '+(', '.join(patterns) if patterns else 'no selected candle pattern'))
    st.write('**Recent observations** (research only; not trade signals):')
    events=aziz_events(x)
    if events.empty: st.caption('No qualifying crosses or engulfing patterns in completed bars.')
    else: st.dataframe(events,hide_index=True,use_container_width=True)
    st.caption('RVOL compares this 5m candle with the preceding 20 completed 5m candles, '
               'not with historical same-time-of-day volume. 15m opening range uses 09:30-09:45 ET. '
               'Patterns are ungraded and must be backtested with spreads, slippage and out-of-sample data.')

# ---------------------------------------------------------------- UI
def _fmt_z(v) -> str:
    return f"{v:+.2f}" if v is not None else "—"


def _extreme_line(e: dict) -> str:
    return (f"{e['time'].strftime('%H:%M')} — "
            f"{'oversold' if e['side'] == 'long' else 'overbought'} "
            f"(z {e['z']:+.2f})")


def render():
    for sym in SYMBOLS:
        try:
            df1 = fetch_bars_1m(sym, _bucket(live, int(refresh_secs)))
        except Exception as e:
            st.error(f"Could not load bars for {sym}: {e}")
            return
        if df1.empty or len(df1) < 60:
            st.info(f"Not enough intraday data for {sym} yet.")
            return
        dates = sorted(set(df1.index.date))
        sel = dates[-1]
        day1 = df1[df1.index.date == sel]
        if len(day1) < 2:
            st.info(f"Not enough bars for {sym} today yet.")
            return
        prev = df1[df1.index.date == dates[-2]] if len(dates) >= 2 else None
        prev_close = float(prev["Close"].iloc[-1]) if prev is not None \
            and len(prev) else None
        df5 = completed_5m(day1)
        if len(df5) < 25:
            st.info(f"Only the first few bars of the {sym} session so "
                    "far — the read builds as bars arrive.")
            return
        read = symbol_read(df5, prev_close)
        st.metric(sym, f"${read['last_close']:,.2f}",
                  f"{read['chg_pct']:+.2f}% vs prior close"
                  if read["chg_pct"] is not None else None)
        st.markdown(
            f"- **z vs 20-bar mean:** {_fmt_z(read['z_mean'])} · "
            f"**z vs session VWAP:** {_fmt_z(read['z_vwap'])} "
            f"(VWAP ${read['vwap']:,.2f}, last bar "
            f"{read['last_bar'].strftime('%H:%M')})")
        reg = read["regime"]
        reg_line = (f"- **Regime (morning autocorr):** {reg['label']}"
                    + (f" (lag-1 autocorr {reg['ac']:+.2f})"
                       if reg["ac"] is not None else "")
                    + (" — forming" if not reg["complete"] else "")
                    + " — the morning label matched the full-session "
                      "label in only 56% of the 41 graded sessions.")
        st.markdown(reg_line)
        ex = read["extremes_mean"][:3]
        if ex:
            st.markdown("- **|z| ≥ 2 today (vs mean):** " + " · ".join(
                _extreme_line(e) for e in ex))
        exv = read["extremes_vwap"][:3]
        if exv:
            st.markdown("- **|z| ≥ 2 today (vs VWAP):** " + " · ".join(
                _extreme_line(e) for e in exv))
        st.plotly_chart(z_chart(sym, read), use_container_width=True,
                        key=f"quant_z_{sym}")
        if show_aziz:
            render_aziz(sym, day1, prev)
        if sym == "SPY":
            spy5, spy_read = df5, read
        else:
            qqq5 = df5

    sp = spread_z(spy5, qqq5)
    sp_z = sp["z"].iloc[-1] if len(sp) else None
    ratio = float(sp["qqq"].iloc[-1] / sp["spy"].iloc[-1]) if len(sp) \
        else None
    st.markdown("**SPY–QQQ spread**")
    st.markdown(
        f"- **Spread z:** {_fmt_z(sp_z if sp_z is not None
                                  and pd.notna(sp_z) else None)} · "
        f"**QQQ/SPY ratio:** {ratio:.4f}" if ratio else
        "- **Spread z:** —")
    sp_ex = z_extremes(sp["z"])[:3]
    if sp_ex:
        st.markdown("- **|spread z| ≥ 2 today:** " + " · ".join(
            _extreme_line(e) for e in sp_ex))
    st.plotly_chart(spread_chart(sp), use_container_width=True,
                    key="quant_spread")
    st.markdown("**Kelly sizer — last graded expectancy "
                f"({GRADED['date']})**")
    rows = []
    for key in ("a1", "a2", "c"):
        g = GRADED[key]
        rows.append(
            f"- **{g['name']}** ({g['horizon']}): n={g['n']}, hit "
            f"{g['hit']:.1f}%, avg {g['avg_gross_bps']:+.1f} bps gross / "
            f"{g['avg_net_bps']:+.1f} bps net.")
    st.markdown("\n".join(rows))
    st.markdown(f"- {kelly_line()}")
    st.caption(
        "Display only — z-scores, regime and spread are context, never "
        "a trigger. Nothing here is graded as a signal: the Nov 19 "
        "grades found no edge in any of the three reads, and the Kelly "
        "sizer reads no size until a new grade finds a positive "
        "expectancy. Quantitative and price-action reads use the same "
        "last completed five-minute bar.")



# -------------------------------------------------------- Consolidated view
# Bulkowski-inspired intraday PROXIES, not exact daily-book patterns.
def bulkowski_intraday(x):
    if len(x)<20:return {'name':'Forming','direction':'neutral','level':None,'note':'Need 20 completed bars'}
    close=x.Close.astype(float); high=x.High.astype(float); low=x.Low.astype(float)
    last=float(close.iloc[-1]); hi=float(high.iloc[-13:-1].max()); lo=float(low.iloc[-13:-1].min())
    atr=float((high-low).iloc[-14:].mean()); tol=max(atr*.55,last*.0005)
    vol=x.RVOL20.iloc[-1] if 'RVOL20' in x else np.nan
    if last>hi and close.iloc[-2]<=high.iloc[-14:-2].max():
        return {'name':'Range breakout up','direction':'up','level':hi,'note':f'Closed above prior 12-bar resistance; RVOL {vol:.2f}x' if pd.notna(vol) else 'Volume not available'}
    if last<lo and close.iloc[-2]>=low.iloc[-14:-2].min():
        return {'name':'Range breakout down','direction':'down','level':lo,'note':f'Closed below prior 12-bar support; RVOL {vol:.2f}x' if pd.notna(vol) else 'Volume not available'}
    if close.iloc[-2]>high.iloc[-14:-2].max() and last<hi:
        return {'name':'Failed upside breakout','direction':'down','level':hi,'note':'Close returned below prior resistance — graded on 5-min QQQ: price re-takes the extreme 73% of the time, so this is not a reliable trap signal'}
    if close.iloc[-2]<low.iloc[-14:-2].min() and last>lo:
        return {'name':'Failed downside breakout','direction':'up','level':lo,'note':'Close returned above prior support — graded on 5-min QQQ: price re-takes the extreme 73% of the time, so this is not a reliable trap signal'}
    # Confirmed swing points are only drawn after two subsequent candles.
    peaks=[i for i in range(2,len(x)-2) if high.iloc[i]>=high.iloc[i-2:i+3].max()]
    valleys=[i for i in range(2,len(x)-2) if low.iloc[i]<=low.iloc[i-2:i+3].min()]
    if len(peaks)>=2 and abs(high.iloc[peaks[-1]]-high.iloc[peaks[-2]])<=tol and peaks[-1]-peaks[-2]>=3:
        neck=float(low.iloc[peaks[-2]:peaks[-1]+1].min())
        return {'name':'Possible double top','direction':'down' if last<neck else 'neutral','level':neck,'note':'Only confirmed below intervening valley (neckline)'}
    if len(valleys)>=2 and abs(low.iloc[valleys[-1]]-low.iloc[valleys[-2]])<=tol and valleys[-1]-valleys[-2]>=3:
        neck=float(high.iloc[valleys[-2]:valleys[-1]+1].max())
        return {'name':'Possible double bottom','direction':'up' if last>neck else 'neutral','level':neck,'note':'Only confirmed above intervening peak (neckline)'}
    width=(hi-lo)/last
    if width<.0045:return {'name':'Tight rectangle / consolidation','direction':'neutral','level':hi,'note':'Wait for a completed close outside range; 0.45% width is an experimental threshold'}
    return {'name':'No confirmed selected pattern','direction':'neutral','level':None,'note':'No high-confidence pattern proxy; do not force a trade'}


# ------------------------------------------------ Bulkowski — graded, display only
# Verdicts from the Nov 20, 2026 grade (grade_bulkowski.py): no rule has
# an edge, so nothing here is a signal. Numbers are quoted exactly as
# graded — net bps/trade after 2 bps cost unless marked gross.
BULKOWSKI_GRADE_ROWS = [
    ("12-bar range breakout", "41", "+2.4 bps", "No edge — halves disagree (IS −2.1 / OOS +6.7)"),
    ("Breakout + flagpole ≥1%", "13", "—", "Untestable — 0 pass, max flagpole 0.46%"),
    ("Breakout + volume", "8 / 11", "−4.3 / +4.8 gross", "Inconclusive — light volume ahead, n too small"),
    ("Failed breakout, trade reversal", "41", "−6.9 bps", "No edge — OOS −11.0 gross"),
    ("Partial decline/rise filter", "1 of 13", "—", "Inconclusive — too rare at this scale"),
    ("Fade the overnight gap", "66", "−3.1 bps", "No edge — fills 25% by close vs book 89–93%/week"),
    ("Follow the overnight gap", "24", "+10.2 bps (+60m)", "Inconclusive — retest candidate, not a signal"),
    ("Measured-move 2nd leg", "1", "—", "Inconclusive — funnel 41 → 5 → 2 → 1"),
    ("Throwback re-entry", "31", "+4.0 bps", "No edge — IS −0.7; +15m retest candidate"),
]


def render_bulkowski_grade():
    with st.expander("Bulkowski — graded Nov 20, 2026 (display only, no signal)", expanded=False):
        st.markdown(
            "| Rule | n | +30m net | Verdict |\n"
            "|---|---:|---:|---|\n"
            + "\n".join(
                f"| {rule} | {n} | {net} | {verdict} |"
                for rule, n, net, verdict in BULKOWSKI_GRADE_ROWS
            )
        )
        st.caption(
            "Descriptive facts, not trades: 76% of 12-bar breakouts return to the breakout level within "
            "12 bars (31/41); after a failed breakout, price re-takes the breakout extreme 73% of the time "
            "before reaching the reversal target (30/41) — the trap usually isn't a trap on 5-min QQQ. "
            "Rectangle measure target is reached first only 15% (6/41) vs the book's 80–85% on daily bars."
        )
        st.caption(
            "Retest candidates — follow the gap +60m and post-throwback +15m — need a fresh sample before "
            "they count for anything. Kelly: NO SIZE for every Bulkowski rule. Double-top proxy tolerance "
            "is ~10× tighter than the book's 0.95% median peak variation, so it fires rarely by design; "
            "widening it is untested. Graded on QQQ 5-min SIP 41 sessions Sep 4–Nov 2 and SPY+QQQ Yahoo "
            "5-min 60 sessions Aug 27–Nov 19; signals 10:00 ET onward, one per session per rule, split-sample."
        )


def weinstein_intraday_context(x: pd.DataFrame) -> dict:
    """Experimental 5-minute phase proxy. NOT Weinstein's weekly stage analysis."""
    if len(x) < 35:
        return {'phase': 'Insufficient history', 'detail': 'Need 35 completed 5-minute bars.'}
    close = x.Close.astype(float)
    ma = close.ewm(span=30, adjust=False).mean()
    atr = (x.High - x.Low).rolling(14).mean().iloc[-1]
    if not np.isfinite(atr) or atr <= 0:
        return {'phase': 'Indeterminate', 'detail': 'Volatility estimate unavailable.'}
    slope = (ma.iloc[-1] - ma.iloc[-7]) / atr
    gap = (close.iloc[-1] - ma.iloc[-1]) / atr
    width = (x.High.iloc[-19:-1].max() - x.Low.iloc[-19:-1].min()) / atr
    # Research heuristics only: never infer a weekly stage from these values.
    if slope > 0.35 and gap > 0.25:
        phase = 'Advancing-like (intraday)'
    elif slope < -0.35 and gap < -0.25:
        phase = 'Declining-like (intraday)'
    elif abs(slope) < 0.35 and width < 5:
        phase = 'Basing / topping-like (intraday)'
    else:
        phase = 'Transition / mixed (intraday)'
    vol = x.RVOL20.iloc[-1] if 'RVOL20' in x else np.nan
    detail = (f'30-bar EMA slope {slope:+.2f} ATR / 6 bars; distance {gap:+.2f} ATR; '
              f'prior 18-bar range {width:.2f} ATR; '
              f'RVOL {vol:.2f}x.' if pd.notna(vol) else
              f'30-bar EMA slope {slope:+.2f} ATR / 6 bars; distance {gap:+.2f} ATR.')
    return {'phase': phase, 'detail': detail}


def _unified_nison_events(x: pd.DataFrame, max_events: int = 7):
    """Completed-bar Nison observations; markers use their observable timestamps."""
    try:
        f = make_features(x)
        definitions = [('nison_engulfing','Engulfing'),('nison_morning_evening','Morning/evening star'),
                       ('nison_hammer','Hammer'),('nison_shooting_star','Shooting star'),
                       ('nison_window','Window/gap'),('nison_reversal_confirm','Price confirmation')]
        events=[]
        for i in range(max(0,len(f)-45),len(f)):
            row=f.iloc[i]
            for field,label in definitions:
                val=row.get(field,0)
                if pd.notna(val) and float(val)!=0:
                    # Hammer and shooting star are directional by definition.
                    direction = (1 if field=='nison_hammer' else -1 if field=='nison_shooting_star' else (1 if val>0 else -1))
                    events.append((f.index[i], label, direction, i))
        return events[-max_events:]
    except Exception:
        return []


def _unified_forecast_panel(sym: str, x: pd.DataFrame, price_bias: str):
    """Compact summary; never equate raw model confidence with calibrated probability."""
    horizon=st.selectbox('Forecast horizon', [5,15,30], index=1, key=f'unified_horizon_{sym}',
                         format_func=lambda n:f'Next {n} minutes')
    if not show_forecast:
        st.info('Enable forward-direction research above to display model estimates.')
        return
    try:
        history=_forward_history(sym,_bucket(live,int(refresh_secs)),45)
        result=evaluate_forward(history,cost_bps=4.0,features=FEATURES)
        r=result.get(horizon//5,{})
        if r.get('status')!='ok':
            st.info('Insufficient historical observations for this forecast.')
            return
        label,reason=_forecast_label(r,price_bias)
        p=float(r['p_up']); confidence=max(p,1-p)
        raw_dir='UP' if p>=.5 else 'DOWN'
        c1,c2,c3=st.columns(3)
        c1.metric('Raw hypothesis',f'{raw_dir} {confidence:.1%}')
        c2.metric('Trade readiness',label)
        c3.metric('Holdout accuracy',f'{r["accuracy"]:.1%}')
        st.caption(f'Uncalibrated model estimate · holdout n={r["n_test"]} · '
                   f'net {r["mean_net_bps"]:+.1f} bps/trade · '
                   f'Brier {r["brier"]:.3f} vs baseline {r["baseline_brier"]:.3f}. {reason}')
        # Simple reliability sample for the SAME forecast confidence band.
        band_low=min(90,int(confidence*10)*10)
        oos=r.get('oos_prob_up',[]); returns=r.get('oos_forward_bps',[])
        pairs=[(float(q),float(y)) for q,y in zip(oos,returns) if pd.notna(q) and pd.notna(y)]
        band=[(q,y) for q,y in pairs if band_low/100 <= max(q,1-q) < (band_low+10)/100]
        if len(band)>=30:
            hits=sum((q>=.5 and y>0) or (q<.5 and y<0) for q,y in band)
            st.caption(f'Similar raw confidence band {band_low}–{band_low+10}%: '
                       f'{hits}/{len(band)} correct ({hits/len(band):.1%}). '
                       'Descriptive holdout frequency, not a calibrated live probability.')
        else:
            st.caption(f'Similar raw confidence band {band_low}–{band_low+10}%: '
                       f'only {len(band)} holdout observations — insufficient reliability evidence.')
        last_end=history.index[-1]+pd.Timedelta(minutes=5)
        if in_market_hours() and (pd.Timestamp.now(tz=ET)-last_end).total_seconds()>600:
            st.warning('Forecast data are stale (>10 minutes).')
    except Exception as exc:
        st.warning(f'Forecast unavailable: {exc}')




def _chart_pattern_observations(x: pd.DataFrame, recent: int = 50):
    """Causal, experimental intraday pattern observations on completed bars.

    Range breaks, consolidations and simple converging ranges are proxies, not
    validated Bulkowski/Edwards-Magee pattern classifications. A pattern is
    timestamped at the bar on which it becomes observable.
    """
    events=[]
    if len(x)<22:
        return events
    for i in range(max(20,len(x)-recent),len(x)):
        window=x.iloc[i-12:i]  # strictly prior completed bars
        current=x.iloc[i]
        hi=float(window.High.max()); lo=float(window.Low.min())
        close=float(current.Close)
        atr=float((window.High-window.Low).mean())
        if not np.isfinite(atr) or atr<=0:
            continue
        vol=float(current.RVOL20) if 'RVOL20' in x and pd.notna(current.RVOL20) else float('nan')
        volnote=f'RVOL {vol:.2f}x' if np.isfinite(vol) else 'RVOL unavailable'
        # A forming range is only shown near the latest bars, not on every bar.
        if close>hi:
            events.append((x.index[i], 'Range breakout UP', 1, i, hi, volnote))
        elif close<lo:
            events.append((x.index[i], 'Range breakdown DOWN', -1, i, lo, volnote))
        elif i>=21:
            previous=x.iloc[i-1]; prior=x.iloc[i-13:i-1]
            prevhi=float(prior.High.max()); prevlo=float(prior.Low.min())
            if float(previous.Close)>prevhi and close<hi:
                events.append((x.index[i], 'Failed upside breakout', -1, i, hi, volnote))
            elif float(previous.Close)<prevlo and close>lo:
                events.append((x.index[i], 'Failed downside breakdown', 1, i, lo, volnote))
    return events[-12:]


def _forming_pattern_context(x: pd.DataFrame):
    """Describe an observable range/triangle candidate without future pivots."""
    if len(x)<26:
        return {'name':'Insufficient bars','detail':'Need 26 completed bars','top':None,'bottom':None}
    w=x.iloc[-18:]
    highs=w.High.astype(float).to_numpy(); lows=w.Low.astype(float).to_numpy()
    axis=np.arange(len(w),dtype=float)
    slope_h=float(np.polyfit(axis,highs,1)[0]); slope_l=float(np.polyfit(axis,lows,1)[0])
    atr=float((w.High-w.Low).mean()); price=float(w.Close.iloc[-1])
    top=float(w.High.max()); bottom=float(w.Low.min())
    if not np.isfinite(atr) or atr<=0:
        return {'name':'Unclassified','detail':'Volatility unavailable','top':top,'bottom':bottom}
    width=(top-bottom)/price
    if slope_h < -0.04*atr and slope_l > 0.04*atr:
        name='Symmetrical triangle candidate'
    elif slope_h < -0.04*atr and abs(slope_l)<0.04*atr:
        name='Descending triangle candidate'
    elif abs(slope_h)<0.04*atr and slope_l>0.04*atr:
        name='Ascending triangle candidate'
    elif slope_h < -0.04*atr and slope_l < -0.04*atr and slope_l>slope_h:
        name='Falling wedge candidate'
    elif slope_h > 0.04*atr and slope_l > 0.04*atr and slope_h<slope_l:
        name='Rising wedge candidate'
    elif width < 0.005:
        name='Rectangle / consolidation candidate'
    else:
        name='No clear forming pattern'
    return {'name':name,'detail':f'18-bar range ${bottom:.2f}–${top:.2f}; width {width:.2%}. Pattern is provisional, not a forecast.',
            'top':top,'bottom':bottom}


def _overlay_chart_patterns(fig, x: pd.DataFrame, show_patterns: bool=True):
    if not show_patterns:
        return fig
    events=_chart_pattern_observations(x)
    for direction,color,symbol,name in [(1,'#059669','star','Bullish structure'),(-1,'#dc2626','star','Bearish structure')]:
        chosen=[e for e in events if e[2]==direction]
        if not chosen:continue
        fig.add_trace(go.Scatter(
            x=[e[0] for e in chosen],
            y=[float(x.iloc[e[3]].Low)-float(x.iloc[e[3]].High-x.iloc[e[3]].Low)*.55 if direction==1
               else float(x.iloc[e[3]].High)+float(x.iloc[e[3]].High-x.iloc[e[3]].Low)*.55 for e in chosen],
            mode='markers',name=name,
            marker=dict(symbol=symbol,color=color,size=11,line=dict(color='white',width=1)),
            text=[f'{e[1]}<br>{e[0].strftime("%H:%M")} ET (bar start)<br>Level ${e[4]:.2f}<br>{e[5]}<br>Experimental pattern observation, not a trading signal' for e in chosen],
            hovertemplate='%{text}<extra></extra>'))
    return fig

def render_consolidated():
    st.markdown("""<style>
    .unified-metric {min-height: 104px; padding: 8px 10px; margin-bottom: 8px;}
    .unified-metric-label {font-size: 0.85rem; line-height: 1.35; color: var(--text-color); opacity: .8;}
    .unified-metric-value {font-size: clamp(1.05rem, 1.65vw, 1.55rem); font-weight: 600;
        line-height: 1.3; overflow-wrap: anywhere; white-space: normal; margin-top: 9px;}
    @media (max-width: 760px) {.unified-metric-value {font-size: 1.05rem;}}
    </style>""", unsafe_allow_html=True)
    st.caption('SPY and QQQ price charts use completed 1-minute candles; the existing predictive models still forecast from completed 5-minute bars. Chart-pattern rules are experimental 1-minute adaptations.')
    show_patterns=st.checkbox('Show chart-pattern markers on SPY and QQQ',value=True,key='show_chart_patterns_both')
    for sym in SYMBOLS:
        st.markdown(f'### {sym} — unified market view')
        try:
            raw=fetch_bars_1m(sym,_bucket(live,int(refresh_secs)))
            if raw.empty:st.warning(f'{sym}: no Alpaca data');continue
            dates=sorted(set(raw.index.date)); day=raw[raw.index.date==dates[-1]]
            prev=raw[raw.index.date==dates[-2]] if len(dates)>1 else None
            x=price_action_features(completed_1m(day),prev,bar_minutes=1)
            if len(x)<26:st.info(f'{sym}: waiting for 26 completed 1-minute candles');continue
            r=x.iloc[-1]; pattern=bulkowski_intraday(x)
            forming=_forming_pattern_context(x)
            ws_context = weinstein_intraday_context(x)
            p=float(r.Close);rv=float(r.RVOL20) if pd.notna(r.RVOL20) else None
            above=p>float(r.VWAP); aligned=float(r.EMA9)>float(r.EMA20)
            state=('Bullish alignment' if above and aligned else 'Bearish alignment' if not above and not aligned else 'Mixed / wait')
            # Compact, wrapping metric cards avoid Streamlit metric ellipsis.
            from html import escape
            metrics = [
                ('Last completed close', f'${p:,.2f}'),
                ('Trend context', state),
                ('1m relative volume', f'{rv:.2f}x' if rv is not None else '—'),
                ('1m pattern proxy', pattern['name']),
            ]
            c = st.columns(4)
            for column, (label, value) in zip(c, metrics):
                with column:
                    st.markdown(
                        '<div class="unified-metric">'
                        f'<div class="unified-metric-label">{escape(label)}</div>'
                        f'<div class="unified-metric-value">{escape(value)}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
            st.caption(f'Forming 1-minute chart pattern: **{forming["name"]}** — {forming["detail"]}')
            st.caption(f'Weinstein intraday proxy: **{ws_context["phase"]}** — {ws_context["detail"]}')
            try:
                weekly = original_weekly_stage(fetch_daily_for_weinstein(sym, _bucket(live, int(refresh_secs))))
                st.caption(f'Original weekly trend context: **{weekly["stage"]}** — {weekly["detail"]}')
            except Exception as weekly_exc:
                st.caption(f'Weekly trend context unavailable: {weekly_exc}')
            # The paired ETF is fetched only for synchronized relative-strength context.
            try:
                other_sym='SPY' if sym=='QQQ' else 'QQQ'
                other_raw=fetch_bars_1m(other_sym,_bucket(live,int(refresh_secs)))
                other_day=other_raw[other_raw.index.date==dates[-1]]
                other_five=completed_5m(other_day)
                own_five=completed_5m(day)
                relative=relative_strength_context(own_five,other_five) if sym=='SPY' else relative_strength_context(other_five,own_five)
                st.caption(relative['text'])
            except Exception as rs_exc:
                st.caption(f'Relative strength unavailable: {rs_exc}')
            st.markdown('#### Unified prediction & confirmation')
            _unified_forecast_panel(sym,x,'Bullish' if state=='Bullish alignment' else 'Bearish' if state=='Bearish alignment' else 'Mixed')
            # Price chart remains a single interactive chart per ETF; event labels are hover-only.
            fig=aziz_chart(sym,x,show_events=True,event_count=4)
            nison_events=_unified_nison_events(x)
            if nison_events:
                for direction, color, symbol, name in [(1,'#059669','diamond','Nison bullish'),
                                                        (-1,'#be123c','diamond','Nison bearish')]:
                    chosen=[e for e in nison_events if e[2]==direction]
                    if not chosen:continue
                    xs=[e[0] for e in chosen]
                    ys=[float(x.iloc[e[3]].Low)-float(x.iloc[e[3]].High-x.iloc[e[3]].Low)*.40 if direction==1
                        else float(x.iloc[e[3]].High)+float(x.iloc[e[3]].High-x.iloc[e[3]].Low)*.40 for e in chosen]
                    texts=[f'{e[1]} · {e[0].strftime("%H:%M")} ET (bar start)<br>'
                           'Experimental Nison observation; not a validated trade signal' for e in chosen]
                    fig.add_trace(go.Scatter(x=xs,y=ys,mode='markers',name=name,
                        marker=dict(color=color,size=10,symbol=symbol,line=dict(color='white',width=1)),
                        text=texts,hovertemplate='%{text}<extra></extra>'))
            _overlay_chart_patterns(fig,x,show_patterns=show_patterns)
            if pattern['level'] is not None:
                fig.add_hline(y=pattern['level'],line_dash='dash',line_color='#b45309',annotation_text=f'Pattern level {pattern["level"]:.2f}',annotation_position='left')
            # Show only two projected structural levels; no invented price forecasts.
            ph=float(x.High.iloc[-13:-1].max());pl=float(x.Low.iloc[-13:-1].min())
            fig.add_hline(y=ph,line_dash='dash',line_color='#0f766e',opacity=.55,annotation_text=f'12m R {ph:.2f}',annotation_position='left')
            fig.add_hline(y=pl,line_dash='dash',line_color='#be123c',opacity=.55,annotation_text=f'12m S {pl:.2f}',annotation_position='left')
            fig.update_layout(title=f'{sym} | 1-minute price action + research observations',height=590,dragmode='zoom',uirevision=f'{sym}-unified',xaxis_rangeslider_visible=True)
            st.plotly_chart(fig,use_container_width=True,key=f'unified_{sym}',config={'scrollZoom':True,'displaylogo':False})
            st.caption(f'1-minute pattern: {pattern["note"]}. EMA9/EMA20 are calculated on 1-minute bars; the forward model remains 5-minute based. EMA9 yellow · EMA20 blue · VWAP purple. Nison diamonds show completed-bar observations; hover for details.')
            if show_forecast:
                with st.expander(f'{sym} — detailed forecasts, probability reliability & model comparisons',expanded=False):
                    render_forward(sym, price_bias=('Bullish' if state == 'Bullish alignment' else 'Bearish' if state == 'Bearish alignment' else 'Mixed'))
            with st.expander(f'{sym} — detailed research and risk checklist',expanded=False):
                render_decision(sym,x)
                st.caption('Z-score / spread studies previously showed no positive net edge; excluded as trading triggers.')
        except Exception as exc:
            st.error(f'{sym}: dashboard could not be calculated: {exc}')
    render_bulkowski_grade()
    st.caption('Weinstein research updated from the accessible original Scribd text (1988), chapters 2-4 and 7-8. Weekly stage is computed from completed weekly bars; intraday phase is separately labeled a proxy. Neither is a validated 5-minute trading signal.')
    st.caption('Chart patterns from Bulkowski were researched mainly on daily historical stocks. This intraday adaptation is not validated for SPY/QQQ. Forecasts are model estimates, not guaranteed price targets. No automated orders.')

st.set_page_config(page_title="Quant Panel — SPY & QQQ", layout="wide")
st.title("Quant Panel — SPY & QQQ")
st.caption("Unified Aziz + Edwards–Magee + Bulkowski + original Weinstein weekly context and relative-strength research. "
           "No validated trading signals or automatic orders.")

show_forecast = st.toggle("Enable forward-direction research", value=True, help="Uses historical Alpaca bars and a chronological holdout; may take time on first load.")
show_aziz = st.toggle("Show Aziz price-action research", value=True,
                           help="Adds unvalidated price-action observations, no orders.")
live = st.toggle("Live tracking (auto-refresh)", value=True,
                 help="When on, the panel re-runs automatically at the "
                      "interval below.")
refresh_secs = st.number_input(
    "Refresh every (seconds)", min_value=30, max_value=3600, value=30,
    step=5,
    help="How often live tracking re-runs the panel. Below 30s the "
         "server runs out of threads/memory.")
if st.button("🔄 Refresh now"):
    try:
        st.session_state["refresh_nonce"] = \
            st.session_state.get("refresh_nonce", 0) + 1
    except Exception:
        pass

if live and not in_market_hours():
    st.caption("Live tracking is on, but the market is closed — showing "
               "the last session without auto-refresh.")


@st.fragment(run_every=int(refresh_secs) if live and in_market_hours()
             else None)
def _show():
    render_consolidated()


_show()
st.caption(
    f"Alpaca {str(_secrets()[2]).upper()} · 5-min bars, full session · "
    f"SPY + QQQ · updated {datetime.now(ET).strftime('%H:%M:%S')} "
    f"ET · build {BUILD_TAG} · {BUILD_HASH}")
