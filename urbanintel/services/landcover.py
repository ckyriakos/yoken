"""Land-cover change on Vas. Olgas from Sentinel-2: did the 2026 redevelopment make the corridor greener?

Same-season (June-September) median NDVI composites for three years: before the 2020 closure, during the works, and the first summer
after the April 2026 planting. Pixels are classed tree / grass / other with the same NDVI thresholds as lulc.yaml.
The corridor (Olgas edges buffered by corridor_buffer_m) is compared with a control ring around it, imaged by the same scenes, so
year-to-year weather and sensor differences cancel: difference-in-differences = change(corridor) - change(control).
The observed change in tree and grass share is translated into an expected surface-temperature change with the land-cover response
model's coefficients (lulc_model_coefficients.csv). Results are cached in data/outputs/cases/kallimarmaro/landcover/."""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

from .. import layers as LY
from . import mobility, overview

OUT = mobility.CASE_DIR / "landcover"
LAYER_IDS = {"olgas_ndvi_pre_works": "Olgas NDVI, before works", "olgas_ndvi_works": "Olgas NDVI, during works",
             "olgas_ndvi_after": "Olgas NDVI, after (2026)", "olgas_ndvi_change": "Olgas NDVI change, after − before"}


def cfg():
    return mobility.cfg()["landcover"]


def _geoms():
    c = cfg()
    olg = gpd.read_file(mobility.CASE_DIR / "olgas.geojson").to_crs(2100) if (mobility.CASE_DIR / "olgas.geojson").exists() \
        else mobility.network()[1].pipe(lambda e: e[e.olgas]).to_crs(2100)
    line = olg.geometry.union_all()
    corridor = line.buffer(c["corridor_buffer_m"])
    lo, hi = c["control_ring_m"]
    control = line.buffer(hi).difference(line.buffer(lo))
    return line, corridor, control


def _grid(corridor, res, margin=450):
    x0, y0, x1, y1 = corridor.bounds
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    w, h = int((x1 - x0) // res), int((y1 - y0) // res)
    return {"crs": "EPSG:2100", "transform": from_origin(x0, y1, res, res), "width": w, "height": h}


def register_layers():
    for lid, title in LAYER_IDS.items():
        tif = OUT / f"{lid}.tif"
        if lid not in LY.BY_ID and tif.exists():
            ch = lid.endswith("change")
            l = LY.Layer(lid, title, "Case: Vas. Olgas", str(tif.relative_to(LY.ROOT)), "raster",
                         "Sentinel-2 L2A June-September median NDVI, 10 m", cmap="BrBG" if ch else "RdYlGn",
                         vrange=(-0.2, 0.2) if ch else (-0.05, 0.65), clip=False)
            LY.LAYERS.append(l); LY.BY_ID[lid] = l


def _shares(ndvi, mask, th):
    v = ndvi[mask & np.isfinite(ndvi)]
    if not len(v):
        return {"ndvi_mean": None, "tree_pct": None, "grass_pct": None, "other_pct": None, "n_px": 0}
    tree = v >= th["tree"]; grass = (v >= th["grass"]) & ~tree
    return {"ndvi_mean": round(float(v.mean()), 4), "tree_pct": round(100 * float(tree.mean()), 2),
            "grass_pct": round(100 * float(grass.mean()), 2), "other_pct": round(100 * float((~tree & ~grass).mean()), 2), "n_px": int(len(v))}


def _lst_coefs():
    try:
        k = pd.read_csv(LY.OUT / "lulc_model_coefficients.csv")
        k = k[k.scope == "local"] if (k.scope == "local").any() else k
        return {t: float(k[k.term == t].estimate.mean()) for t in ("tree", "grass")}
    except Exception:
        return {"tree": -3.0, "grass": 0.0}


def run(force=False, log=print):
    """Fetch (cached per scene/band), composite, class and compare. Needs network access the first time."""
    from pipeline import s2fetch
    p = OUT / "landcover.json"
    if p.exists() and not force:
        register_layers()
        return json.loads(p.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    c = cfg()
    line, corridor, control = _geoms()
    grid = _grid(corridor, c["resolution_m"])
    shp = (grid["height"], grid["width"])
    m_cor = rasterize([(corridor, 1)], out_shape=shp, transform=grid["transform"], fill=0, dtype="uint8").astype(bool)
    m_ctl = rasterize([(control, 1)], out_shape=shp, transform=grid["transform"], fill=0, dtype="uint8").astype(bool)
    bbox = s2fetch.grid_bbox_wgs84(grid)
    prof = dict(driver="GTiff", dtype="float32", count=1, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                height=grid["height"], nodata=np.nan, compress="deflate")
    comps, periods = {}, {}
    for key, year in c["periods"].items():
        live = {"year": year, "start": f"{year}-{min(c['months']):02d}-01", "end": f"{year}-{max(c['months']):02d}-30",
                "months": c["months"], "max_cloud_pct": c["max_cloud_pct"], "max_scenes": c["max_scenes"]}
        items = s2fetch.search_scenes(bbox, live)
        if not items:
            log(f"  {key} ({year}): no scenes"); continue
        log(f"  {key} ({year}): {len(items)} scenes")
        out, nv = s2fetch.composite_indices(items, ["ndvi"], grid, min_valid=2, cache_dir=LY.ROOT / "data" / "cache", log=lambda *_: None)
        comps[key] = out["ndvi"].astype("float32")
        with rasterio.open(OUT / f"olgas_ndvi_{key}.tif", "w", **prof) as dst:
            dst.write(comps[key], 1)
        periods[key] = {"year": year, "scenes": s2fetch.scene_table(items),
                        "corridor": _shares(comps[key], m_cor, c["thresholds"]), "control": _shares(comps[key], m_ctl, c["thresholds"])}
    if "pre_works" in comps and "after" in comps:
        ch = comps["after"] - comps["pre_works"]
        with rasterio.open(OUT / "olgas_ndvi_change.tif", "w", **prof) as dst:
            dst.write(ch.astype("float32"), 1)
    coefs = _lst_coefs()

    def did(a, b):
        if a not in periods or b not in periods:
            return None
        res = {}
        for k in ("ndvi_mean", "tree_pct", "grass_pct"):
            ca, cb = periods[a]["corridor"][k], periods[b]["corridor"][k]
            ka, kb = periods[a]["control"][k], periods[b]["control"][k]
            if None in (ca, cb, ka, kb):
                continue
            res[k] = {"corridor": round(cb - ca, 4), "control": round(kb - ka, 4), "did": round((cb - ca) - (kb - ka), 4)}
        if "tree_pct" in res and "grass_pct" in res:
            res["expected_lst_change_c"] = round(coefs["tree"] * res["tree_pct"]["did"] / 100 + coefs["grass"] * res["grass_pct"]["did"] / 100, 3)
        return res
    result = {"periods": periods, "change": {"after_vs_pre_works": did("pre_works", "after"), "after_vs_works": did("works", "after"),
                                             "works_vs_pre_works": did("pre_works", "works")},
              "corridor_area_ha": round(corridor.area / 1e4, 2), "control_area_ha": round(control.area / 1e4, 2),
              "settings": c, "lst_coefficients_c_per_unit_share": coefs,
              "caveat": "10 m pixels mix street, pavement and canopy; trees planted in April 2026 are young, so the canopy signal is small "
                        "and grows over the next years. Difference-in-differences uses the control ring imaged by the same scenes."}
    p.write_text(json.dumps(result, indent=2, default=str))
    gpd.GeoDataFrame({"zone": ["corridor", "control"]}, geometry=[corridor, control], crs=2100).to_crs(4326) \
        .to_file(OUT / "zones.geojson", driver="GeoJSON")
    register_layers()
    for lid in LAYER_IDS:
        if (OUT / f"{lid}.tif").exists():
            overview.prepare_layer(lid, force=True, log=log)
    return result


def zones():
    p = OUT / "zones.geojson"
    return json.loads(p.read_text()) if p.exists() else None


register_layers()
