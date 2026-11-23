"""Sentinel-2 land cover on a 10 m grid aligned to the Landsat 30 m grid (each 30 m pixel = exactly 3x3 cells).

Classes (hard, per 10 m cell):  ground (paved or bare, anything not vegetated or built) | tree | grass | building.
  building = footprint covers >= t_bld of the cell (footprints, not Sentinel-2)
  tree     = NDVI >= t_tree      grass = t_grass <= NDVI < t_tree      ground = the rest
Thresholds are assumptions (see lulc.yaml); the scenario runner repeats everything with low/base/high thresholds.
Sentinel-2 at 10 m cannot see single street trees, understorey, or the difference between bare soil and asphalt."""
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter

GROUND, TREE, GRASS, BUILDING, NODATA = 0, 1, 2, 3, 255
CLASS_NAMES = ["ground", "tree", "grass", "building"]
FRAC_NAMES = ["tree", "grass", "building"]          # ground is the reference class (shares sum to 1)


def grid10(profile):
    """10 m grid inside a north-up 30 m profile (same origin, 3x the rows and columns)."""
    t = profile["transform"]
    if t.b != 0 or t.d != 0 or t.e >= 0:
        raise ValueError("expected a north-up raster")
    return {"transform": Affine(t.a / 3, 0, t.c, 0, t.e / 3, t.f), "crs": profile["crs"],
            "height": profile["height"] * 3, "width": profile["width"] * 3}


def building_cover(geoms, grid, ss=2):
    """Share of each 10 m cell covered by footprints (rasterised at 10/ss m, then averaged)."""
    H, W = grid["height"], grid["width"]
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros((H, W), "float32")
    a = rasterize(shapes, out_shape=(H * ss, W * ss), transform=grid["transform"] @ Affine.scale(1 / ss),
                  fill=0, dtype="uint8")
    return a.reshape(H, ss, W, ss).mean(axis=(1, 3)).astype("float32")


def classify(ndvi, bld, t_tree, t_grass, t_bld=0.5):
    """Hard class codes per 10 m cell. Cells with no valid NDVI and no building stay NODATA."""
    if t_grass >= t_tree:
        raise ValueError("t_grass must be below t_tree")
    ok = np.isfinite(ndvi)
    cls = np.full(ndvi.shape, GROUND, "uint8")
    cls[ok & (ndvi >= t_grass)] = GRASS
    cls[ok & (ndvi >= t_tree)] = TREE
    cls[~ok] = NODATA
    cls[bld >= t_bld] = BUILDING
    return cls


def to_state(cls):
    """One-hot shares (4, h, w) in the order ground, tree, grass, building, plus a validity mask."""
    valid = cls != NODATA
    S = np.stack([(cls == k) for k in (GROUND, TREE, GRASS, BUILDING)]).astype("float32")
    return S, valid


def state_fractions(S, valid, min_valid=0.5):
    """30 m shares (tree, grass, building) from 10 m state. Shares are of the valid part of each 30 m pixel;
    pixels with less than min_valid valid area are NaN. h and w must be multiples of 3."""
    _, h, w = S.shape
    H, W = h // 3, w // 3
    v = valid.reshape(H, 3, W, 3).mean(axis=(1, 3))
    F = np.stack([S[k].reshape(H, 3, W, 3).mean(axis=(1, 3)) for k in (TREE, GRASS, BUILDING)])
    F = F / np.maximum(v, 1e-9)
    F[:, v < min_valid] = np.nan
    return F.astype("float32"), v.astype("float32")


def focal_mean(a, size):
    """NaN-aware moving mean over size x size pixels on the last two axes (3-D or 2-D input).
    Pixels where less than half of the window is valid are NaN."""
    a = np.asarray(a, "float64")
    s = (1,) * (a.ndim - 2) + (size, size)
    ok = np.isfinite(a)
    num = uniform_filter(np.where(ok, a, 0.0), size=s, mode="constant")
    den = uniform_filter(ok.astype("float64"), size=s, mode="constant")
    return np.where(den >= 0.5, num / np.maximum(den, 1e-12), np.nan).astype("float32")


def class_shares(cls, mask=None):
    """Percent of valid cells per class (optionally inside a boolean mask)."""
    c = cls[mask] if mask is not None else cls.ravel()
    c = c[c != NODATA]
    n = max(len(c), 1)
    return {name: round(100 * float((c == k).sum()) / n, 2) for k, name in enumerate(CLASS_NAMES)}
