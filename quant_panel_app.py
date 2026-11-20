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
BUILD_TAG = "2026-11-20-bulkowski-graded-v4"

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


def price_action_features(df5: pd.DataFrame, prev: pd.DataFrame | None = None,
                          opening_minutes: int = 15) -> pd.DataFrame:
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
    n = max(1, opening_minutes // 5)
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




# -------------------------------------------------------- forward direction research
"""Walk-forward directional research for 5-minute SPY/QQQ bars.
No orders. Probabilities are uncalibrated model estimates, with held-out diagnostics.
Includes Edwards-Magee-inspired causal structural features.
"""

BASE_FEATURES = ['ema_gap','vwap_gap','momentum3','momentum6','rvol','range_pct','structure','or_position','z20','macd_hist']
MAGEE_FEATURES = ['trend_efficiency','swing_position','resistance_distance','support_distance',
                  'breakout_up','breakout_down','failed_up','failed_down',
                  'compression','wick_pressure','volume_confirmation']
FEATURES = BASE_FEATURES + MAGEE_FEATURES
HORIZONS = (1,3,6)


def make_features(bars):
    """Input: complete regular-session 5m bars, possibly multiple sessions."""
    if bars.empty:
        return pd.DataFrame()
    chunks=[]
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


def evaluate_forward(bars, cost_bps=4.0, min_train=150, min_test=40, features=None):
    """Chronological 70/30 held-out evaluation, with embargo per horizon.
    Latest inference trains only on rows with realized outcomes available.
    Results are descriptive and NOT calibrated probabilities.
    """
    f=make_features(bars)
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
                'cost_bps':cost_bps}
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


def render_consolidated():
    st.markdown("""<style>
    .unified-metric {min-height: 104px; padding: 8px 10px; margin-bottom: 8px;}
    .unified-metric-label {font-size: 0.85rem; line-height: 1.35; color: var(--text-color); opacity: .8;}
    .unified-metric-value {font-size: clamp(1.05rem, 1.65vw, 1.55rem); font-weight: 600;
        line-height: 1.3; overflow-wrap: anywhere; white-space: normal; margin-top: 9px;}
    @media (max-width: 760px) {.unified-metric-value {font-size: 1.05rem;}}
    </style>""", unsafe_allow_html=True)
    st.caption('One interactive chart per ETF. All annotations use completed 5-minute candles. Bulkowski pattern rules here are experimental intraday adaptations.')
    for sym in SYMBOLS:
        st.markdown(f'### {sym} — unified market view')
        try:
            raw=fetch_bars_1m(sym,_bucket(live,int(refresh_secs)))
            if raw.empty:st.warning(f'{sym}: no Alpaca data');continue
            dates=sorted(set(raw.index.date)); day=raw[raw.index.date==dates[-1]]
            prev=raw[raw.index.date==dates[-2]] if len(dates)>1 else None
            x=price_action_features(completed_5m(day),prev)
            if len(x)<20:st.info(f'{sym}: waiting for 20 completed 5-minute candles');continue
            r=x.iloc[-1]; pattern=bulkowski_intraday(x)
            p=float(r.Close);rv=float(r.RVOL20) if pd.notna(r.RVOL20) else None
            above=p>float(r.VWAP); aligned=float(r.EMA9)>float(r.EMA20)
            state=('Bullish alignment' if above and aligned else 'Bearish alignment' if not above and not aligned else 'Mixed / wait')
            # Compact, wrapping metric cards avoid Streamlit metric ellipsis.
            from html import escape
            metrics = [
                ('Last completed close', f'${p:,.2f}'),
                ('Trend context', state),
                ('Relative volume', f'{rv:.2f}x' if rv is not None else '—'),
                ('Bulkowski pattern proxy', pattern['name']),
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
            fig=aziz_chart(sym,x,show_events=True,event_count=5)
            if pattern['level'] is not None:
                fig.add_hline(y=pattern['level'],line_dash='dash',line_color='#b45309',annotation_text=f'Pattern level {pattern["level"]:.2f}',annotation_position='left')
            # Show only two projected structural levels; no invented price forecasts.
            ph=float(x.High.iloc[-13:-1].max());pl=float(x.Low.iloc[-13:-1].min())
            fig.add_hline(y=ph,line_dash='dash',line_color='#0f766e',opacity=.55,annotation_text=f'12-bar R {ph:.2f}',annotation_position='left')
            fig.add_hline(y=pl,line_dash='dash',line_color='#be123c',opacity=.55,annotation_text=f'12-bar S {pl:.2f}',annotation_position='left')
            fig.update_layout(title=f'{sym} | Price + EMA9 / EMA20 + VWAP + structure',height=590,dragmode='zoom',uirevision=f'{sym}-unified',xaxis_rangeslider_visible=True)
            st.plotly_chart(fig,use_container_width=True,key=f'unified_{sym}',config={'scrollZoom':True,'displaylogo':False})
            st.caption(f'Pattern: {pattern["note"]}. EMA9 yellow · EMA20 blue · VWAP purple. Zoom using mouse wheel or drag; double-click to reset.')
            if show_forecast:
                with st.expander(f'{sym} — 5 / 15 / 30 minute forecasts and out-of-sample checks',expanded=True):
                    render_forward(sym, price_bias=('Bullish' if state == 'Bullish alignment' else 'Bearish' if state == 'Bearish alignment' else 'Mixed'))
            with st.expander(f'{sym} — detailed research and risk checklist',expanded=False):
                render_decision(sym,x)
                st.caption('Z-score / spread studies previously showed no positive net edge; excluded as trading triggers.')
        except Exception as exc:
            st.error(f'{sym}: dashboard could not be calculated: {exc}')
    render_bulkowski_grade()
    st.caption('Chart patterns from Bulkowski were researched mainly on daily historical stocks. This intraday adaptation is not validated for SPY/QQQ. Forecasts are model estimates, not guaranteed price targets. No automated orders.')

st.set_page_config(page_title="Quant Panel — SPY & QQQ", layout="wide")
st.title("Quant Panel — SPY & QQQ")
st.caption("Unified Aziz + Edwards–Magee + Bulkowski pattern research. "
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
