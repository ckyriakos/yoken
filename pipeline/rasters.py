"""Raster helpers: Landsat LST in deg C, Sentinel-2 NDVI aggregated onto the Landsat grid."""
import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

QA_BAD_BITS = sum(1 << b for b in range(6))      # fill, dilated cloud, cirrus, cloud, shadow, snow
LST_MIN_C, LST_MAX_C = 0.0, 70.0                  # plausibility mask for Athens in summer


def lst_celsius(st_path, qa_path, scale=0.00341802, offset=149.0):
    """Return (lst_c, profile, qc). Invalid pixels are NaN. qc counts what was masked."""
    with rasterio.open(st_path) as s, rasterio.open(qa_path) as q:
        dn, qa, prof = s.read(1).astype("float64"), q.read(1), s.profile
    clear = (qa & QA_BAD_BITS) == 0
    lst = np.where(dn > 0, dn * scale + offset - 273.15, np.nan)
    out_of_range = np.isfinite(lst) & ((lst < LST_MIN_C) | (lst > LST_MAX_C))
    lst[out_of_range | ~clear] = np.nan
    qc = {"pixels": int(dn.size), "qa_masked": int((~clear & (dn > 0)).sum()),
          "out_of_range": int(out_of_range.sum()), "valid": int(np.isfinite(lst).sum())}
    return lst.astype("float32"), prof, qc


def ndvi_on_grid(red_path, nir_path, scl_path, ref_profile, red_scale=1e-4, red_offset=-0.1,
                 nir_scale=1e-4, nir_offset=-0.1, bad_scl=(0, 1, 3, 6, 7, 8, 9, 10, 11)):
    """NDVI at native 10 m (after SCL mask), averaged onto the reference (Landsat) grid.
    Returns (ndvi_30m, valid_fraction, qc). Scale/offset should come from the STAC item."""
    with rasterio.open(red_path) as r, rasterio.open(nir_path) as n, rasterio.open(scl_path) as s:
        red = r.read(1).astype("float32") * red_scale + red_offset
        nir = n.read(1).astype("float32") * nir_scale + nir_offset
        scl = s.read(1, out_shape=red.shape, resampling=Resampling.nearest)
        src_t, src_crs = r.transform, r.crs
    ok = ~np.isin(scl, bad_scl) & (red > 0) & (nir > 0)
    raw = np.where(ok, (nir - red) / (nir + red + 1e-12), 0).astype("float32")
    qc = {"clipped_gt1": int((raw > 1).sum()), "clipped_lt_m1": int((raw < -1).sum()),
          "masked_nonpositive_refl": int(((red <= 0) | (nir <= 0)).sum())}
    qc["median_red_refl"] = float(np.median(red[ok])) if ok.any() else float("nan")
    qc["share_nonpositive_pct"] = float(100 * ((red <= 0) | (nir <= 0)).mean())
    if qc["share_nonpositive_pct"] > 5:
        warnings.warn(f"{qc['share_nonpositive_pct']:.1f}% non-positive reflectance: check S2 offset")
    ndvi = np.clip(raw, -1, 1)
    shape = (ref_profile["height"], ref_profile["width"])
    num, frac = np.zeros(shape, "float32"), np.zeros(shape, "float32")
    for src, dst in ((ndvi * ok, num), (ok.astype("float32"), frac)):
        reproject(src, dst, src_transform=src_t, src_crs=src_crs,
                  dst_transform=ref_profile["transform"], dst_crs=ref_profile["crs"],
                  resampling=Resampling.average)
    out = np.where(frac > 0.5, num / np.where(frac > 0, frac, 1), np.nan)   # need >50% valid 10 m px
    return out.astype("float32"), frac, qc


def composite(stack, min_valid):
    """Pixel-wise median over scenes; NaN where fewer than min_valid scenes are valid."""
    n_valid = np.isfinite(stack).sum(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN pixels are expected
        med = np.nanmedian(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    med[n_valid < min_valid] = np.nan
    return med.astype("float32"), n_valid.astype("int16")
