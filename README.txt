Quant Panel V22.1 + V22.2
========================
Files:
  quant_panel_spy_qqq_v22.py   - main Streamlit app (rename to app.py)
  v22_validation.py            - rolling walk-forward research evaluation
  v22_ledger.py                - SQLite forecast ledger

Deployment:
1. Place ALL THREE Python files together in the same folder / GitHub repository.
2. Rename quant_panel_spy_qqq_v22.py to app.py.
3. Retain the existing requirements.txt and Alpaca secrets from V21.
4. Set QUANT_LEDGER_PATH to a writable PERSISTENT filesystem path if one is available.
   Without persistent storage, SQLite survives Streamlit reruns but may be lost on redeploy/restart.
5. Open SPY or QQQ Forward Direction Engine. The app now reports walk-forward
   folds, a session-bootstrap uncertainty interval, and 1/2/4 bps cost sensitivity.
6. View the timestamped forecast outcomes expander and export the ledger to CSV.

Notes:
- Rolling 20 training sessions / 5 testing sessions, 5/15/30 minute horizons.
- Evaluation uses nonoverlapping test predictions within sessions, and only
  completed labels. No live orders or automatic trades are placed.
- Conservative 42-test family-wise adjustment for uncertainty intervals.
- Thresholds are intentionally conservative; 'NO EDGE' is expected until evidence improves.
- Full Streamlit and live Alpaca tests are not possible in this environment.
- This version does not change Alpaca incremental fetching, regularization tuning,
  or feature pruning; those remain future engineering tasks.
