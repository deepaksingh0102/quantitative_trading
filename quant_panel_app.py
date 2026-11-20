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
BUILD_TAG = "2026-11-19"

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
        df5 = resample_5min(day1)
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
        "expectancy. The last 5-min bar is still forming, so its "
        "z-score is provisional.")


st.set_page_config(page_title="Quant Panel — SPY & QQQ", layout="wide")
st.title("Quant Panel — SPY & QQQ")
st.caption("Chan, Quantitative Trading — the graded reads, live. "
           "Display only: no signals, no verdict.")

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
    render()


_show()
st.caption(
    f"Alpaca {str(_secrets()[2]).upper()} · 5-min bars, full session · "
    f"SPY + QQQ · updated {datetime.now(ET).strftime('%H:%M:%S')} "
    f"ET · build {BUILD_TAG} · {BUILD_HASH}")
