"""Layer preparation and serving: one code path for the UI and the API.

prepare_all() is run once (python -m urbanintel.prepare); it writes light, web-ready copies of every catalogue layer to
data/outputs/demo/layers/: vectors as EPSG:4326 GeoJSON clipped to the municipality and simplified, rasters as EPSG:3857 COGs plus a
coloured PNG overlay and its WGS84 bounds."""
from __future__ import annotations

import json
from functools import lru_cache
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from shapely.geometry import box

from .. import layers as LY

AOI_WGS84 = (23.68, 37.93, 23.80, 38.05)          # same as config.yaml aoi_bbox_wgs84
CLIP_BUFFER_M = 300
MAX_PNG_PX = 2048


# ---------------------------------------------------------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def municipality(crs=2100):
    d = gpd.read_file(LY.ROOT / LY.BY_ID["districts"].path).to_crs(2100)
    return d.geometry.make_valid().union_all().buffer(CLIP_BUFFER_M)


def _read_vector(layer):
    g = gpd.read_file(LY.ROOT / layer.path)
    if g.crs is None:
        g = g.set_crs(4326)
    return g


def _prepare_vector(layer):
    g = _read_vector(layer).to_crs(2100)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if layer.kind in ("polygon",):
        g["geometry"] = g.geometry.make_valid()
    if layer.clip:
        area = municipality()
        g = g[g.intersects(area)]
        if layer.kind in ("polygon", "line"):
            g["geometry"] = g.geometry.intersection(area)
            g = g[~g.geometry.is_empty]
    if layer.simplify_m:
        g["geometry"] = g.geometry.simplify(layer.simplify_m, preserve_topology=True)
    if layer.kind in ("point", "heat"):
        g["geometry"] = g.geometry.representative_point() if not (g.geom_type == "Point").all() else g.geometry
    keep = [c for c in layer.keep if c in g.columns] if layer.keep else [c for c in g.columns if c != "geometry"]
    g = g[keep + ["geometry"]].to_crs(4326)
    for c in keep:                                  # JSON-safe attributes
        if not pd.api.types.is_numeric_dtype(g[c]):
            g[c] = g[c].astype(str).replace({"None": None, "nan": None, "NaT": None})
    return g


def _prepare_raster(layer):
    src_path = LY.ROOT / layer.path
    cog = LY.LAYER_DIR / f"{layer.id}.tif"
    with rasterio.open(src_path) as src:
        nod = src.nodata
        if layer.clip:
            b = transform_bounds("EPSG:4326", src.crs, *AOI_WGS84)
            win = rasterio.windows.from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            arr = src.read(1, window=win).astype("float32")
            tr = src.window_transform(win)
        else:
            arr, tr = src.read(1).astype("float32"), src.transform
        if nod is not None:
            arr[arr == nod] = np.nan
        h, w = arr.shape
        bounds = rasterio.transform.array_bounds(h, w, tr)
        dt, dw, dh = calculate_default_transform(src.crs, "EPSG:3857", w, h, *bounds)
        scale = max(1.0, max(dw, dh) / MAX_PNG_PX)
        dw, dh = int(dw / scale), int(dh / scale)
        dt = dt * dt.scale(scale, scale)
        out = np.full((dh, dw), np.nan, "float32")
        cat = layer.categories is not None
        reproject(arr, out, src_transform=tr, src_crs=src.crs, dst_transform=dt, dst_crs="EPSG:3857",
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest if cat else Resampling.bilinear)
    prof = dict(driver="COG", dtype="float32", count=1, height=dh, width=dw, crs="EPSG:3857", transform=dt, nodata=np.nan,
                compress="deflate")
    with rasterio.open(cog, "w", **prof) as dst:
        dst.write(out, 1)
    rgba, vmin, vmax = colourise(out, layer)
    Image.fromarray(rgba, "RGBA").save(LY.LAYER_DIR / f"{layer.id}.png", optimize=True)
    wb = transform_bounds("EPSG:3857", "EPSG:4326", *rasterio.transform.array_bounds(dh, dw, dt))
    meta = {"bounds": [[wb[1], wb[0]], [wb[3], wb[2]]], "vmin": vmin, "vmax": vmax, "cmap": layer.cmap, "categories": layer.categories}
    (LY.LAYER_DIR / f"{layer.id}.json").write_text(json.dumps(meta))
    return meta


def colourise(arr, layer, alpha=200):
    """float array -> RGBA uint8 with NaN transparent. Categorical layers: 0 also transparent."""
    finite = np.isfinite(arr)
    if layer.vrange:
        vmin, vmax = layer.vrange
    else:
        vmin, vmax = (np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)) if finite.any() else (0, 1)
    norm = np.clip((np.nan_to_num(arr, nan=vmin) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = (colormaps[layer.cmap](norm) * 255).astype("uint8")
    a = np.where(finite, alpha, 0)
    if layer.categories is not None:
        a = np.where(finite & np.isin(arr, list(layer.categories)), alpha, 0)
        if layer.vrange == (0, 1):                  # binary masks: single strong colour
            rgba = (colormaps[layer.cmap](np.where(arr == 1, 0.85, 0)) * 255).astype("uint8")
    rgba[..., 3] = a
    return rgba, float(vmin), float(vmax)


def prepare_layer(layer_id, force=False, log=print):
    layer = LY.get(layer_id)
    LY.LAYER_DIR.mkdir(parents=True, exist_ok=True)
    p = LY.prepared_path(layer)
    if p.exists() and not force:
        return p
    if layer.kind == "raster":
        _prepare_raster(layer)
    else:
        g = _prepare_vector(layer)
        g.to_file(p, driver="GeoJSON", COORDINATE_PRECISION=6)
    log(f"  {layer.id:20s} -> {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
    return p


def prepare_all(force=False, log=print):
    for l in LY.LAYERS:
        try:
            prepare_layer(l.id, force, log)
        except Exception as e:      # one broken source must not block the demo
            log(f"  {l.id:20s} FAILED: {e}")


# ---------------------------------------------------------------------------------------------------------------- serving
@lru_cache(maxsize=64)
def _vector(layer_id):
    layer = LY.get(layer_id)
    p = prepare_layer(layer_id)
    return gpd.read_file(p) if layer.kind != "raster" else None


def vector(layer_id, bbox=None, simplify=None, limit=None) -> gpd.GeoDataFrame:
    """Prepared layer as a GeoDataFrame (EPSG:4326), optionally clipped to bbox=(minx, miny, maxx, maxy) in WGS84."""
    layer = LY.get(layer_id)
    if layer.kind == "raster":
        raise ValueError(f"{layer_id} is a raster; use raster_png or tile")
    if layer_id in LY.BBOX_ONLY and bbox is None:
        raise ValueError(f"{layer_id} is large; pass a bbox")
    g = _vector(layer_id)
    if bbox is not None:
        g = g[g.intersects(box(*bbox))]
    if simplify:
        g = g.copy()
        g["geometry"] = g.geometry.simplify(simplify)
    if limit:
        g = g.head(limit)
    return g


def geojson(layer_id, **kw) -> dict:
    g = vector(layer_id, **kw)
    dt = [c for c in g.columns if c != "geometry" and pd.api.types.is_datetime64_any_dtype(g[c])]
    if dt:
        g = g.assign(**{c: g[c].astype(str) for c in dt})
    return json.loads(g.to_json(drop_id=True))


def heat_points(layer_id, bbox=None):
    """[[lat, lon, weight], ...] for heatmap layers."""
    g = vector(layer_id, bbox=bbox)
    layer = LY.get(layer_id)
    w = g[layer.weight].astype(float).to_numpy() if layer.weight and layer.weight in g else np.ones(len(g))
    return np.c_[g.geometry.y, g.geometry.x, w].round(6).tolist()


def raster_meta(layer_id) -> dict:
    prepare_layer(layer_id)
    return json.loads((LY.LAYER_DIR / f"{layer_id}.json").read_text())


def raster_png(layer_id) -> bytes:
    return prepare_layer(layer_id).read_bytes()


def tile(layer_id, z, x, y) -> bytes | None:
    """256 px XYZ tile (PNG) of a prepared raster, coloured like the overlay. None if the tile is outside the layer."""
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader
    layer = LY.get(layer_id)
    prepare_layer(layer_id)
    try:
        with Reader(str(LY.LAYER_DIR / f"{layer_id}.tif")) as r:
            img = r.tile(x, y, z, tilesize=256, resampling_method="nearest")
    except TileOutsideBounds:
        return None
    arr = img.data[0].astype("float32")
    arr[img.mask == 0] = np.nan
    rgba, *_ = colourise(arr, layer)
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return buf.getvalue()


def catalogue():
    return [l.to_dict() for l in LY.LAYERS]


# ---------------------------------------------------------------------------------------------------------------- headline
# ---------------------------------------------------------------------------------------------------------------- districts
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th"}


@lru_cache(maxsize=1)
def _districts():
    d = gpd.read_file(LY.ROOT / LY.BY_ID["districts"].path).to_crs(2100)
    d["geometry"] = d.geometry.make_valid()
    d["dk"] = d.dk.astype(int)
    return d


@lru_cache(maxsize=1)
def districts() -> dict:
    """{dk: label} for the 7 municipal districts (Δημοτικές Κοινότητες); label lists their largest areas."""
    d = _districts().assign(a=lambda x: x.area)
    out = {}
    for dk, g in d.groupby("dk"):
        names = [str(n).split("-")[0].strip().title() for n in g.sort_values("a", ascending=False).name_en.head(3)]
        out[int(dk)] = f"{ORD.get(int(dk), dk)}: {', '.join(names)}"
    return dict(sorted(out.items()))


def norm_districts(dks):
    """None / [] / all seven -> () meaning 'whole municipality'; otherwise a sorted tuple of district numbers."""
    if not dks:
        return ()
    t = tuple(sorted({int(x) for x in dks}))
    return () if set(t) >= set(districts()) else t


@lru_cache(maxsize=16)
def district_geom(dks=()):
    """Union of the selected districts (EPSG:2100); the whole municipality for ()."""
    d = _districts()
    if dks:
        d = d[d.dk.isin(dks)]
    return d.geometry.union_all()


@lru_cache(maxsize=16)
def nb_ids(dks=()):
    """Neighbourhood ids in the selected districts (all for ())."""
    ind = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")
    if dks:
        ind = ind[ind.district_id.astype(int).isin(dks)]
    return tuple(ind.neighbourhood_id)


@lru_cache(maxsize=16)
def headline(dks=()) -> dict:
    """Four numbers for the overview strip, for the whole city or the selected districts. Each falls back to None."""
    dks = norm_districts(dks)
    ids = set(nb_ids(dks))
    out = {}
    try:
        bp = pd.read_csv(LY.OUT / "building_population.csv", usecols=["pop_cal", "neighbourhood_id"])
        out["population"] = float(bp[bp.neighbourhood_id.isin(ids)].pop_cal.sum())
    except Exception:
        out["population"] = None
    try:
        ind = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")
        ind = ind[ind.neighbourhood_id.isin(ids)]
        out["lst_p90_mean_c"] = float(ind.lst_p90_c.mean())
        out["hotspot_share_pct"] = float(ind.hotspot_share_pct.mean())
    except Exception:
        out["lst_p90_mean_c"] = out["hotspot_share_pct"] = None
    try:
        ac = pd.read_csv(LY.OUT / "neighbourhood_access.csv")
        ac = ac[ac.neighbourhood_id.isin(ids)]
        out["green_within5_pct"] = float((ac.green_open_within5_pct * ac.population).sum() / ac.population.sum())
    except Exception:
        out["green_within5_pct"] = None
    try:
        from . import flood
        ex = flood.exposure_by_neighbourhood()
        ppl = ex[(ex.zone == "flood_100") & ex.neighbourhood_id.isin(ids)].people.sum()
        out["flood100_pop_pct"] = float(100 * ppl / out["population"]) if out.get("population") else None
    except Exception:
        out["flood100_pop_pct"] = None
    out["districts"] = list(dks)
    return out
