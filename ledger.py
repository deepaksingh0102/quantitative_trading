"""Persistent SQLite forecast/outcome ledger; deploy with a persistent disk.
Do not retroactively create predictions from historical prices.
"""
import os
import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd

SCHEMA = '''CREATE TABLE IF NOT EXISTS forecasts (
    symbol TEXT NOT NULL, bar_start TEXT NOT NULL, horizon_bars INTEGER NOT NULL,
    model_version TEXT NOT NULL, training_cutoff TEXT NOT NULL,
    recorded_at TEXT NOT NULL, start_price REAL NOT NULL, p_up REAL NOT NULL,
    hypothesis TEXT NOT NULL, display TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING', end_price REAL, move_bps REAL,
    correct INTEGER, settled_at TEXT,
    PRIMARY KEY(symbol, bar_start, horizon_bars, model_version)
)'''


def ledger_path():
    return os.environ.get('QUANT_LEDGER_PATH', 'quant_forecasts.sqlite3')


def _connect(path=None):
    path = str(path or ledger_path())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15)
    conn.execute('PRAGMA busy_timeout=15000')
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def record_and_settle(symbol, results, five, label_fn, in_market_hours,
                      now=None, path=None, model_version='v22.1-walkforward-l2'):
    if five.empty:
        return []
    now = now or pd.Timestamp.now(tz='America/New_York')
    if pd.Timestamp(now).tzinfo is None:
        raise ValueError('now must be timezone-aware')
    latest_ts = five.index[-1]
    last_close = float(five.Close.iloc[-1])
    with _connect(path) as db:
        pending = db.execute('SELECT symbol,bar_start,horizon_bars,model_version,start_price,hypothesis '
                             'FROM forecasts WHERE symbol=? AND status=?', (symbol, 'PENDING')).fetchall()
        for sym, start_str, h, ver, start_price, hypothesis in pending:
            start = pd.Timestamp(start_str)
            target = start + pd.Timedelta(minutes=5*h)
            if target.date() != start.date():
                db.execute('UPDATE forecasts SET status=?,settled_at=? WHERE symbol=? AND bar_start=? AND horizon_bars=? AND model_version=?',
                           ('EXPIRED', now.isoformat(), sym, start_str, h, ver))
            elif target in five.index and latest_ts >= target:
                end_price = float(five.loc[target, 'Close'])
                move = (end_price/start_price - 1)*10000
                actual = 'UP' if move > 0 else 'DOWN' if move < 0 else 'FLAT'
                db.execute('UPDATE forecasts SET status=?,end_price=?,move_bps=?,correct=?,settled_at=? '
                           'WHERE symbol=? AND bar_start=? AND horizon_bars=? AND model_version=?',
                           (actual, end_price, move, int(actual == hypothesis), now.isoformat(),
                            sym, start_str, h, ver))
            elif latest_ts.date() > start.date():
                db.execute('UPDATE forecasts SET status=?,settled_at=? WHERE symbol=? AND bar_start=? AND horizon_bars=? AND model_version=?',
                           ('EXPIRED', now.isoformat(), sym, start_str, h, ver))
        age = (now - (latest_ts + pd.Timedelta(minutes=5))).total_seconds()/60
        if in_market_hours() and -1 <= age <= 2:
            for h, r in results.items():
                if r.get('status') != 'ok':
                    continue
                p = float(r['p_up'])
                display = label_fn(r)[0]
                db.execute('INSERT OR IGNORE INTO forecasts '
                           '(symbol,bar_start,horizon_bars,model_version,training_cutoff,recorded_at,start_price,p_up,hypothesis,display) '
                           'VALUES(?,?,?,?,?,?,?,?,?,?)',
                           (symbol, latest_ts.isoformat(), int(h), model_version,
                            r.get('train_cutoff','unknown'), now.isoformat(), last_close, p,
                            'UP' if p >= .5 else 'DOWN', display))
        db.commit()
        rows = db.execute('SELECT bar_start,horizon_bars,model_version,training_cutoff,recorded_at,start_price,p_up,hypothesis,display,status,move_bps,correct '
                          'FROM forecasts WHERE symbol=? ORDER BY recorded_at DESC LIMIT 300', (symbol,)).fetchall()
    keys = ('bar_start','h','model_version','training_cutoff','recorded_at','start_price','p_up',
            'hypothesis','display','actual','move_bps','correct')
    return [dict(zip(keys, r)) for r in rows]
