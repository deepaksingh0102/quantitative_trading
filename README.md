# Quant Panel — SPY & QQQ (display only)

Chan, *Quantitative Trading*, graded first and then built as a live
panel. The three reads were graded on his data (Nov 19, 2026) and
**none of them has an edge as a signal**, so this app has no signals,
no verdict and no Buy/Sell. It shows the same numbers live, as
context.

## What it shows

- **Per symbol (SPY, QQQ):** live z-score vs the 20-bar mean and
  z-score vs session VWAP, on 5-min bars — hindsight-free, exactly
  the graded definition (mean/std from bars closed before the signal
  bar). The day's |z| ≥ 2 events are listed with times, de-duplicated
  the way the grade did it (first fresh cross, re-arm inside ±0.5).
- **Morning regime label:** lag-1 autocorrelation of 5-min log
  returns, 09:30–10:30 — mean-reverting / trending / random walk.
  The morning label matched the full-session label in only 56% of
  the 41 graded sessions, so it is a hint, not a classification.
- **SPY–QQQ spread z-score:** log(QQQ) − log(SPY), rolling 20 bars,
  reset each session, as graded.
- **Kelly sizer:** fed by the last graded expectancy (A1 +30m,
  n=87, edge −1.05 bps/trade) — Kelly f ≤ 0, so it reads **no size**
  until a new grade finds a positive edge.

## The grades (Nov 19, 2026)

| Read | Result |
|---|---|
| A1 z vs 20-bar mean (QQQ 5-min, 41 sessions Sep 4–Nov 2) | +30m hit 52.9%, avg −0.9 bps gross / −2.9 net. In-sample +30m hit 61.4% faded to 44.2% out-of-sample. No edge. |
| A2 z vs session VWAP (same file) | +30m hit 60.6%, avg +0.3 / −1.7 net — hit rate without edge. No edge. |
| B regime label | Morning label matched the full-session label only 56%; fades lost on average in every regime. |
| C SPY–QQQ spread z (Yahoo 5-min, 60 sessions Aug 27–Nov 19) | +30m hit 59.1%, avg +1.8 bps gross / −2.2 net — held out-of-sample, but +1–2 bps gross vs 4 bps two-leg costs. No edge. |
| Kelly (A1 +30m) | p=0.529, avg win 10.6 bps, avg loss 14.2 bps, edge −1.05 bps/trade → f ≤ 0, no bet. |

Grade scripts: `quant_grade/grade_chan.py`, `quant_grade/grade_spread.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run quant_panel_app.py
```

## Deploy (Streamlit Cloud)

1. Create a **new public** GitHub repo (e.g. `quant-panel`) and put
   `quant_panel_app.py` in it.
2. Rename `quant_panel_requirements.txt` to `requirements.txt` and
   put it in the repo root. (This file is `quant_panel_README.md` —
   rename it to `README.md` if you want it in the repo.)
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New
   app** → pick the repo and `quant_panel_app.py` → **Deploy**.
4. Add your Alpaca keys under the app's **Settings → Secrets** as
   TOML — the same Secrets as the price-action charting app:

```toml
APCA_API_KEY_ID = "your-key-id"
APCA_API_SECRET_KEY = "your-secret-key"
ALPACA_FEED = "sip"
```

Never commit real keys to the repo. Keys live in Secrets only.

## Data

Alpaca SIP 1-minute bars, resampled to 5 minutes, regular session
only (09:30–16:00 ET). Live tracking auto-refreshes every 30s during
US market hours and goes static after the close, showing the last
session. The last 5-min bar is still forming, so its z-score is
provisional.

## Self-check

```bash
python test_quant_panel.py   # 32 checks, headless + kaleido render
```

## Build

2026-11-19 · e62c61 (footer shows the build tag and hash — check it
after deploying).
