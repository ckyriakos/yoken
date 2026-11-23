"""Neighbourhood statistics, block-bootstrap CIs, priority score and rank intervals."""
import numpy as np
import pandas as pd


def normalise(x, method="minmax"):
    """Normalise along axis 0 (neighbourhoods); x may be 1-D or (n, B)."""
    lo, hi = (np.nanmin(x, 0), np.nanmax(x, 0)) if method == "minmax" else \
             (np.nanpercentile(x, 5, 0), np.nanpercentile(x, 95, 0))
    return np.clip((x - lo) / np.where(hi > lo, hi - lo, 1), 0, 1)


def score(p90, ndvi, w_heat, method="minmax", eligible=None):
    """priority = 100*(w*H + (1-w)*V); H = normalised P90 LST, V = normalised NDVI deficit.
    Normalisation uses eligible neighbourhoods only. w_heat may be scalar or (B,)."""
    p90, ndvi = np.array(p90, float), np.array(ndvi, float)
    if eligible is not None:
        p90[~eligible], ndvi[~eligible] = np.nan, np.nan
    H = normalise(p90, method)
    V = normalise(-ndvi, method)
    return 100 * (w_heat * H + (1 - w_heat) * V)


def observed_stats(pix, min_pixels, min_blocks, min_cov):
    rows = []
    for i, d in pix.items():
        n_tot = len(d["lst"])
        lv, nv = d["lst"][np.isfinite(d["lst"])], d["ndvi"][np.isfinite(d["ndvi"])]
        cov_l, cov_n = 100 * len(lv) / max(n_tot, 1), 100 * len(nv) / max(n_tot, 1)
        blocks = len(np.unique(d["blk"][np.isfinite(d["lst"])])) if len(lv) else 0
        rows.append(dict(
            idx=i, n_pixels=n_tot, lst_valid_px=len(lv), lst_mean_c=lv.mean() if len(lv) else np.nan,
            lst_median_c=np.median(lv) if len(lv) else np.nan,
            lst_p90_c=np.percentile(lv, 90) if len(lv) else np.nan,
            lst_coverage_pct=cov_l, ndvi_mean=nv.mean() if len(nv) else np.nan,
            ndvi_median=np.median(nv) if len(nv) else np.nan, ndvi_coverage_pct=cov_n, n_blocks=blocks))
    df = pd.DataFrame(rows).set_index("idx")
    df["eligible"] = ((df.lst_valid_px >= min_pixels) & (df.n_blocks >= min_blocks) &
                      (df.lst_coverage_pct >= min_cov) & (df.ndvi_coverage_pct >= min_cov))
    return df


def bootstrap(pix, eligible_idx, n_boot, seed):
    """Block bootstrap within each eligible neighbourhood. Returns P90 LST and mean NDVI, each (n, B)."""
    rng = np.random.default_rng(seed)
    n = len(pix)
    P, N = np.full((n, n_boot), np.nan), np.full((n, n_boot), np.nan)
    for i in eligible_idx:
        d = pix[i]
        ub, inv = np.unique(d["blk"], return_inverse=True)
        members = [np.flatnonzero(inv == k) for k in range(len(ub))]
        for b in range(n_boot):
            pick = rng.integers(0, len(ub), len(ub))
            ix = np.concatenate([members[k] for k in pick])
            l, v = d["lst"][ix], d["ndvi"][ix]
            l, v = l[np.isfinite(l)], v[np.isfinite(v)]
            if len(l): P[i, b] = np.percentile(l, 90)
            if len(v): N[i, b] = v.mean()
    return P, N


def uncertainty_table(P, N, elig, cfg, seed):
    """CIs on P90/NDVI/score and rank intervals, propagating bootstrap AND weight uncertainty."""
    rng = np.random.default_rng(seed + 1)
    B = P.shape[1]
    w = rng.uniform(*cfg["uncertainty"]["weight_heat_range"], B)
    S = score(P, N, w, cfg["scoring"]["normalisation"], eligible=elig)
    Sr = np.where(np.isnan(S), -np.inf, S)
    rank = (-Sr).argsort(0).argsort(0) + 1.0
    rank[~elig] = np.nan
    lo, hi = cfg["uncertainty"]["ci"]
    pc = lambda a, q: np.nanpercentile(a, q, axis=1)
    out = pd.DataFrame({
        "lst_p90_lo": pc(P, lo), "lst_p90_hi": pc(P, hi), "ndvi_mean_lo": pc(N, lo), "ndvi_mean_hi": pc(N, hi),
        "score_lo": pc(S, lo), "score_hi": pc(S, hi),
        "rank_lo": pc(rank, lo), "rank_median": pc(rank, 50), "rank_hi": pc(rank, hi),
        "p_top10": np.nanmean(rank <= 10, axis=1), "p_top_decile": np.nanmean(rank <= max(1, elig.sum() // 10), axis=1)})
    out.loc[~elig] = np.nan
    return out
