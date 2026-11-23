"""Getis-Ord Gi* hotspot detection on the composite LST raster, with Benjamini-Hochberg control."""
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import norm


def gi_star(lst, radius_px=3, q=0.05):
    """Returns (z, hot) where hot is bool (z>0 and BH-adjusted p<q). Edges/NaNs use available neighbours."""
    valid = np.isfinite(lst)
    x = np.where(valid, lst, 0.0)
    n = valid.sum()
    xbar = x[valid].mean()
    s = np.sqrt((x[valid] ** 2).mean() - xbar ** 2)
    yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    k = (xx ** 2 + yy ** 2 <= radius_px ** 2).astype(float)       # binary circular kernel incl. self
    W = convolve(valid.astype(float), k, mode="constant")          # sum w_ij (= sum w_ij^2, binary)
    num = convolve(x, k, mode="constant") - xbar * W
    den = s * np.sqrt(np.maximum((n * W - W ** 2) / (n - 1), 1e-12))
    z = np.where(valid, num / den, np.nan)
    p = np.where(valid, 1 - norm.cdf(z), np.nan)
    pv = p[valid]
    order = np.argsort(pv)
    m = len(pv)
    passed = pv[order] <= q * (np.arange(1, m + 1) / m)
    thr = pv[order][np.flatnonzero(passed).max()] if passed.any() else -1
    hot = valid & (z > 0) & (p <= thr)
    return z.astype("float32"), hot
