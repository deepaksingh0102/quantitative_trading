"""Session-based walk-forward validation for Quant Panel V22.
No execution signal is implied by model probabilities. All evaluation observations
are chosen before outcomes are inspected. Training labels must be fully known
before the start of each test fold.
"""
import math
import numpy as np
import pandas as pd

MODEL_VERSION = 'v22.1-walkforward-l2'
COST_SCENARIOS = (1.0, 2.0, 4.0)
# Conservative family-wise allowance for 2 ETFs x 3 horizons x 7 feature sets.
FAMILY_TESTS = 42


def _session_bootstrap_ci(values, sessions, alpha=.05, draws=1200, seed=20261122):
    """Resample whole sessions, not individual overlapping intraday trades."""
    values = np.asarray(values, dtype=float)
    sessions = np.asarray(sessions)
    unique = np.unique(sessions)
    if len(unique) < 5 or not len(values):
        return (float('nan'), float('nan'))
    sums = np.asarray([values[sessions == d].sum() for d in unique], dtype=float)
    counts = np.asarray([np.sum(sessions == d) for d in unique], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return tuple(float(v) for v in np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def evaluate_walkforward(bars, features, make_features, fit, standardize,
                         horizons=(1, 3, 6), benchmark=None,
                         cost_bps=4.0, train_sessions=20, test_sessions=5,
                         min_train_rows=150, min_test_obs=30,
                         family_tests=FAMILY_TESTS):
    f = make_features(bars, benchmark=benchmark)
    if f.empty:
        return {}
    features = list(features)
    out = {}
    for h in horizons:
        target = f[f'y{h}']
        history = f.loc[f[features].notna().all(axis=1) & target.notna()].copy()
        latest = f.iloc[-1]
        days = sorted(set(history.index.date))
        if len(days) < train_sessions + test_sessions or not latest[features].notna().all():
            out[h] = {'status': 'insufficient', 'n': len(history), 'n_sessions': len(days),
                      'required_sessions': train_sessions + test_sessions}
            continue
        fold_records = []
        folds = 0
        # Rolling training window and strictly later, non-overlapping test sessions.
        for end in range(train_sessions, len(days), test_sessions):
            train_days = set(days[end - train_sessions:end])
            test_days = set(days[end:end + test_sessions])
            if not test_days:
                continue
            train = history[[d in train_days for d in history.index.date]]
            test = history[[d in test_days for d in history.index.date]]
            # The label for a 5m bar is known only after h subsequent bars close.
            # This is conservative even though make_features already prevents cross-session labels.
            cutoff = pd.Timestamp(min(test_days), tz=history.index.tz)
            train = train[train.index + pd.Timedelta(minutes=5 * (h + 1)) <= cutoff]
            if len(train) < min_train_rows or test.empty:
                continue
            xtr, xt = standardize(train[features].to_numpy(float), test[features].to_numpy(float))
            ytr = (train[f'y{h}'] > 0).astype(float).to_numpy()
            try:
                w = fit(xtr, ytr)
            except (np.linalg.LinAlgError, ValueError):
                continue
            p = 1 / (1 + np.exp(-np.clip(xt @ w, -30, 30)))
            actual = test[f'y{h}'].to_numpy(float)
            # Deterministic, non-overlapping evaluation within each session.
            for day, group in test.groupby(test.index.date):
                locs = test.index.get_indexer(group.index)[::h]
                for loc in locs:
                    fold_records.append((str(test.index[loc]), str(day), float(p[loc]), float(actual[loc]),
                                         float(ytr.mean()), folds))
            folds += 1
        if len(fold_records) < min_test_obs or folds < 2:
            out[h] = {'status': 'insufficient', 'n': len(history),
                      'n_test': len(fold_records), 'n_folds': folds,
                      'reason': 'Need at least two test folds and sufficient nonoverlapping observations'}
            continue
        ts, sessions, probabilities, moves, base_rates, fold_ids = map(list, zip(*fold_records))
        probs = np.asarray(probabilities)
        moves = np.asarray(moves)
        signs = np.where(probs >= .5, 1, -1)
        gross = signs * moves
        net = gross - float(cost_bps)
        accuracy = float(np.mean(gross > 0))
        brier = float(np.mean((probs - (moves > 0)) ** 2))
        base_brier = float(np.mean((np.asarray(base_rates) - (moves > 0)) ** 2))
        alpha = .05 / max(1, int(family_tests))
        lo, hi = _session_bootstrap_ci(net, sessions, alpha=alpha)
        scenarios = {}
        for c in COST_SCENARIOS:
            a, b = _session_bootstrap_ci(gross - c, sessions, alpha=alpha)
            scenarios[str(int(c))] = {'mean_net_bps': float(np.mean(gross - c)),
                                      'ci_low': a, 'ci_high': b}
        # Live fit: only outcomes already realized; history has no unknown labels.
        full = history[features].to_numpy(float)
        xfull, xlatest = standardize(full, latest[features].to_numpy(float).reshape(1, -1))
        wf = fit(xfull, (history[f'y{h}'] > 0).astype(float).to_numpy())
        current = float(1 / (1 + np.exp(-np.clip((xlatest @ wf)[0], -30, 30))))
        out[h] = {'status': 'ok', 'p_up': current, 'p_down': 1-current,
                  'n_train': len(history), 'n_test': len(net), 'n_folds': folds,
                  'n_test_sessions': len(set(sessions)), 'accuracy': accuracy,
                  'mean_net_bps': float(np.mean(net)), 'net_ci_low': lo, 'net_ci_high': hi,
                  'brier': brier, 'baseline_brier': base_brier,
                  'up_base_rate': float(np.mean(base_rates)),
                  'last_bar': str(f.index[-1]), 'last_close': float(latest.Close),
                  'train_cutoff': str(history.index[-1]),
                  'test_start': ts[0], 'test_end': ts[-1], 'cost_bps': float(cost_bps),
                  'cost_scenarios': scenarios, 'multiple_tests': int(family_tests),
                  'oos_prob_up': probabilities, 'oos_forward_bps': moves,
                  'model_version': MODEL_VERSION,
                  'test_folds': folds,
                  'evaluation_protocol': 'rolling-session-walkforward; session-block bootstrap; Bonferroni'}
    return out
