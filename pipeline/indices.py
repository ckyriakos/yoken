"""Selectable spectral indices from Sentinel-2 L2A surface reflectance (0..1, after scale and offset).
Each entry: (bands needed, function). Add a line to INDICES to offer a new index in the app and in lulc.yaml.
Bands: B02 blue, B03 green, B04 red, B08 NIR, B11 SWIR1, B12 SWIR2."""
import numpy as np


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def _evi(b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6 * b["B04"] - 7.5 * b["B02"] + 1)


INDICES = {
    "ndvi":   (("B04", "B08"), lambda b: _nd(b["B08"], b["B04"]), "Vegetation greenness (NIR vs red)"),
    "savi":   (("B04", "B08"), lambda b: 1.5 * (b["B08"] - b["B04"]) / (b["B08"] + b["B04"] + 0.5), "Soil-adjusted vegetation index, L = 0.5"),
    "evi":    (("B02", "B04", "B08"), _evi, "Enhanced vegetation index"),
    "ndbi":   (("B08", "B11"), lambda b: _nd(b["B11"], b["B08"]), "Built-up index (SWIR vs NIR)"),
    "mndwi":  (("B03", "B11"), lambda b: _nd(b["B03"], b["B11"]), "Modified water index (green vs SWIR)"),
    "ndwi":   (("B03", "B08"), lambda b: _nd(b["B03"], b["B08"]), "Water / canopy moisture (green vs NIR)"),
    "bsi":    (("B02", "B04", "B08", "B11"), lambda b: _nd(b["B11"] + b["B04"], b["B08"] + b["B02"]), "Bare soil index"),
    "nbr2":   (("B11", "B12"), lambda b: _nd(b["B11"], b["B12"]), "Normalised burn ratio 2 (SWIR1 vs SWIR2): moisture and material"),
    # Shortwave broadband albedo from Sentinel-2 bands (Bonafoni and Sekertekin 2020, linear fit; no BRDF or atmospheric correction beyond L2A)
    "albedo": (("B02", "B03", "B04", "B08", "B11", "B12"),
               lambda b: 0.2266 * b["B02"] + 0.1236 * b["B03"] + 0.1573 * b["B04"] + 0.3417 * b["B08"] + 0.1170 * b["B11"] + 0.0338 * b["B12"],
               "Shortwave albedo (how reflective the surface is)"),
    "brightness": (("B02", "B03", "B04"), lambda b: (b["B02"] + b["B03"] + b["B04"]) / 3, "Visible brightness"),
}


RANGE = {"ndvi": (-1, 1), "savi": (-1.5, 1.5), "evi": (-1, 2), "ndbi": (-1, 1), "mndwi": (-1, 1), "ndwi": (-1, 1), "bsi": (-1, 1),
         "nbr2": (-1, 1), "albedo": (0, 1), "brightness": (0, 1)}


def bands_needed(names):
    need = {"SCL"}
    for n in names:
        need.update(INDICES[n][0])
    return sorted(need)


def compute(names, bands):
    """bands: dict band -> reflectance array. Returns dict name -> float32 array; non-finite values become NaN."""
    out = {}
    for n in names:
        if n not in INDICES:
            raise ValueError(f"unknown index '{n}'. Available: {', '.join(INDICES)}")
        req, fn, _ = INDICES[n]
        missing = [r for r in req if r not in bands]
        if missing:
            raise ValueError(f"index {n} needs bands {missing}")
        v = np.asarray(fn(bands), "float32")
        v[~np.isfinite(v)] = np.nan
        lo, hi = RANGE[n]
        out[n] = np.where(np.isfinite(v), np.clip(v, lo, hi), np.nan).astype("float32")
    return out
