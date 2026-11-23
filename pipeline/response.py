"""Land-cover -> surface-temperature response, fitted on our own 30 m pixels.

    LST_p = b0 + b_tree*f_tree + b_grass*f_grass + b_bld*f_bld + smooth trend(x, y) + e_p

f_* are shares of the 90 m (3x3 pixel) neighbourhood of pixel p, which matches the ~100 m thermal sensor footprint; ground
(paved or bare) is the reference class. A coefficient is the observed surface-temperature difference between a place that is
100% that class and a place that is 100% ground, with the broad spatial trend held fixed.
This is an OBSERVATIONAL association from one summer composite. It is not a causal effect of planting, and it applies to
mature cover only. Intervals come from a block bootstrap (resampling spatial blocks), which respects spatial autocorrelation."""
import numpy as np

N_FRAC = 3                                           # tree, grass, building
COEF_NAMES = ["intercept", "tree", "grass", "building", "x", "y", "x2", "xy", "y2"]


def trend_coords(x, y, centre, scale_m=2000.0):
    return (np.asarray(x) - centre[0]) / scale_m, (np.asarray(y) - centre[1]) / scale_m


def design(Ff, xs, ys):
    """Ff: (3, n) focal shares; xs, ys: scaled coordinates (n,)."""
    return np.column_stack([np.ones(Ff.shape[1]), Ff.T, xs, ys, xs * xs, xs * ys, ys * ys])


def _block_stats(X, y, blk):
    """Per-block X'X and X'y so that bootstrap and CV need no refitting of raw rows."""
    order = np.argsort(blk, kind="stable")
    Xs, ys, bs = X[order], y[order], blk[order]
    ids, start = np.unique(bs, return_index=True)
    xtx = np.add.reduceat(Xs[:, :, None] * Xs[:, None, :], start, axis=0)
    xty = np.add.reduceat(Xs * ys[:, None], start, axis=0)
    return ids, xtx, xty


def fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def block_bootstrap(X, y, blk, n_boot, seed):
    """Coefficient draws (n_boot, p) from resampling whole spatial blocks with replacement."""
    ids, xtx, xty = _block_stats(X, y, blk)
    rng = np.random.default_rng(seed)
    nb, p = len(ids), X.shape[1]
    out = np.empty((n_boot, p))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        A = np.tensordot(w, xtx, 1) + 1e-9 * np.eye(p)
        out[b] = np.linalg.solve(A, w @ xty)
    return out


def block_cv(X, y, blk, k=5, seed=0, trend_cols=(0, 4, 5, 6, 7, 8)):
    """Out-of-block skill: k-fold over spatial blocks. Returns dict with R2 and RMSE of the full model and of a
    trend-only model (what the land-cover terms add)."""
    ids = np.unique(blk)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(ids, rng.permutation(len(ids)) % k))
    fold = np.array([fold_of[b] for b in blk])
    pred, pred0 = np.empty_like(y), np.empty_like(y)
    for f in range(k):
        tr, te = fold != f, fold == f
        pred[te] = X[te] @ fit(X[tr], y[tr])
        pred0[te] = X[te][:, trend_cols] @ fit(X[tr][:, trend_cols], y[tr])
    sst = float(((y - y.mean()) ** 2).sum())
    return {"r2_full": 1 - float(((y - pred) ** 2).sum()) / sst, "rmse_full_c": float(np.sqrt(((y - pred) ** 2).mean())),
            "r2_trend_only": 1 - float(((y - pred0) ** 2).sum()) / sst,
            "rmse_trend_only_c": float(np.sqrt(((y - pred0) ** 2).mean())), "folds": k, "n_blocks": int(len(ids))}


def summarise_coefs(beta_hat, draws, ci=(5, 95)):
    lo, hi = np.percentile(draws, ci, axis=0)
    return [{"term": n, "estimate": float(b), f"p{ci[0]:02d}": float(l), f"p{ci[1]:02d}": float(h)}
            for n, b, l, h in zip(COEF_NAMES, beta_hat, lo, hi)]


def sign_problems(draws_frac):
    """Blocking plausibility gate: trees must not be warmer than paved or bare ground. Returns messages (empty = ok)."""
    med = np.median(draws_frac, axis=0)
    return [f"tree coefficient is {med[0]:+.2f} C (expected below 0)"] if med[0] >= 0 else []


def sign_warnings(draws_frac, ci=(5, 95)):
    """Non-blocking: grass need not be cooler than ground (dry or sparse cover on hills, mixed pixels). Reported, not enforced."""
    lo, hi = np.percentile(draws_frac[:, 1], ci)
    med = float(np.median(draws_frac[:, 1]))
    if med >= 0 or hi >= 0 and lo <= 0:
        return [f"grass coefficient is {med:+.2f} C [{lo:+.2f}, {hi:+.2f}]: grass is not measurably cooler than paved or bare ground here, "
                "so scenarios that add or remove grass carry little or uncertain temperature effect"]
    return []
