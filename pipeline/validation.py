"""Ground-sensor validation: station air temperature vs Landsat LST at neighbourhood level."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TEMP_MIN, TEMP_MAX = -5.0, 50.0       # plausibility range for air temperature (deg C)
OUTLIER_C = 5.0                       # drop stations this far from the all-station median (and >4 robust SD)


def to_utc_naive(x):
    t = pd.Timestamp(x)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def overpass_buckets(utc):
    """Station timestamps are ceil-to-hour UTC: bucket H holds measurements in (H-1h, H].
    Use the bucket containing the overpass and the one before it (measurements about +-1 h)."""
    c = to_utc_naive(utc).ceil("h")
    return [c - pd.Timedelta(hours=1), c]


def station_temps(obs, buckets):
    """Per-station mean air temperature/humidity over the buckets, with range and outlier QC.
    Returns (table, qc dict)."""
    ts = pd.to_datetime(obs.interval_timestamp)
    ts = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
    d = obs[ts.isin(buckets)]
    qc = {"rows": len(d), "stations_raw": int(d.station_id.nunique())}
    d = d[d.temperature.between(TEMP_MIN, TEMP_MAX)]
    g = d.groupby("station_id").agg(temp_c=("temperature", "mean"), humidity=("humidity", "mean"),
                                    n_buckets=("temperature", "size")).reset_index()
    if len(g) >= 5:
        med = g.temp_c.median()
        rsd = 1.4826 * (g.temp_c - med).abs().median()
        out = (g.temp_c - med).abs() > max(OUTLIER_C, 4 * rsd)
        qc["outliers_dropped"] = int(out.sum())
        g = g[~out]
    else:
        qc["outliers_dropped"] = 0
    qc["stations_used"] = len(g)
    return g, qc


def neighbourhood_temps(st, xwalk):
    """xwalk: df[station_id, neighbourhood_id]. Mean across stations in a neighbourhood plus spread."""
    m = st.merge(xwalk, on="station_id", how="inner")
    return m.groupby("neighbourhood_id").agg(
        temp_c=("temp_c", "mean"), humidity=("humidity", "mean"),
        n_stations=("station_id", "nunique"), temp_sd=("temp_c", "std")).reset_index()


def boot_spearman(x, y, n_boot=2000, seed=0, ci=(5, 95), min_n=8):
    """Spearman rho with a bootstrap CI over neighbourhoods (the unit of comparison)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < min_n:
        return dict(n=n, rho=np.nan, lo=np.nan, hi=np.nan, slope=np.nan, bias_c=np.nan)
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    r = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        r[b] = spearmanr(x[i], y[i])[0] if len(np.unique(x[i])) > 1 and len(np.unique(y[i])) > 1 else np.nan
    return dict(n=n, rho=rho, lo=np.nanpercentile(r, ci[0]), hi=np.nanpercentile(r, ci[1]),
                slope=np.polyfit(x, y, 1)[0], bias_c=float(np.mean(y - x)))   # x = air temp, y = LST
