"""Case study C: flood exposure.

1. Official hazard zones (Flood Directive, T = 50 / 100 / 1000 years) x modelled residents (building_population.csv) -> people and
   buildings exposed per neighbourhood.
2. Network impact: street edges inside the 100-year zone are removed and walking access to bus stops and parks is recomputed for every
   resident (population-weighted, same method as pipeline.network.summarise_access).
3. Satellite evidence: wet-season Sentinel-2 composite (Planetary Computer) -> NDWI, MNDWI, NDVI, NDBI, combined with soil groups and
   distance to streams into a simple runoff-susceptibility score (0-1). This is a screening proxy, not a hydraulic model.
Outputs are cached in data/outputs/cases/flood/ and the rasters are registered as extra map layers (s2_*, runoff)."""
from __future__ import annotations

import json
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from pipeline import abm, network
from .. import layers as LY
from . import overview

CFG_PATH = LY.ROOT / "urbanintel" / "config" / "cases.yaml"
CASE_DIR = LY.OUT / "cases" / "flood"
SPEED = 1.2
CUTOFF_M = 1500
# hydrological response by WRB soil group (higher = more runoff): clay-rich, shrink-swell or shallow soils score high
SOIL_RUNOFF = {"Vertisols": 1.0, "Fluvisols": 0.8, "Leptosols": 0.8, "Luvisols": 0.6, "Cambisols": 0.5, "Calcisols": 0.4,
               "Regosols": 0.4, "Kastanozems": 0.3, "Phaeozems": 0.3}
SAT_LAYERS = {"s2_ndwi": ("Water index, NDWI (wet season Jan–Apr 2026)", "RdBu", (-0.6, 0.2)),
              "s2_mndwi": ("Water index, MNDWI (wet season Jan–Apr 2026)", "RdBu", (-0.6, 0.2)),
              "s2_ndbi": ("Built-up index, NDBI (wet season Jan–Apr 2026)", "OrRd", (-0.3, 0.3)),
              "s2_ndvi": ("Greenness, NDVI (wet season Jan–Apr 2026)", "RdYlGn", (-0.1, 0.7)),
              "runoff": ("Runoff susceptibility (screening)", "YlOrBr", (0.2, 0.8))}


def cfg():
    return yaml.safe_load(open(CFG_PATH))["flood"]


RUNS_DIR = CASE_DIR / "runs"
LOCAL_KEY = "local"


def run_key(start, end):
    return f"{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}"


def layer_id(base, key=LOCAL_KEY):
    """s2_mndwi (local wet season) or s2_mndwi__20251001_20251231 (user-chosen period)."""
    return base if key == LOCAL_KEY else f"{base}__{key}"


def run_dir(key):
    return CASE_DIR if key == LOCAL_KEY else RUNS_DIR / key


def period_label(meta):
    d = sorted(x["datetime"][:10] for x in meta.get("scenes", []))
    if meta.get("start"):
        rng = f"{pd.Timestamp(meta['start']):%-d %b %Y} – {pd.Timestamp(meta['end']):%-d %b %Y}"
    else:
        rng = "wet season Jan–Apr 2026"
    return f"{rng} ({len(d)} images)" if d else rng


def periods():
    """Available Sentinel-2 periods: the local wet season first, then user-requested ranges (newest first)."""
    out = []
    p = CASE_DIR / "spectral.json"
    if p.exists():
        m = json.loads(p.read_text())
        out.append({"key": LOCAL_KEY, "label": "Local: " + period_label(m), "status": "done", **m})
    if RUNS_DIR.exists():
        rs = []
        for d in RUNS_DIR.iterdir():
            f = d / "spectral.json"
            if f.exists():
                m = json.loads(f.read_text())
                rs.append({"key": d.name, "label": period_label(m), **m})
        out += sorted(rs, key=lambda r: r.get("requested_at", ""), reverse=True)
    return out


def register_layers():
    """Add the satellite-derived rasters (local and every finished custom period) to the layer catalogue (idempotent)."""
    for per in periods():
        if per.get("status") != "done":
            continue
        key = per["key"]
        for base, (title, cmap, vr) in SAT_LAYERS.items():
            lid = layer_id(base, key)
            tif = run_dir(key) / f"{base}.tif"
            if lid in LY.BY_ID or not tif.exists():
                continue
            t = title if key == LOCAL_KEY else title.split(" (")[0] + f" ({period_label(per)})"
            l = LY.Layer(lid, t, "Hazards" if key == LOCAL_KEY else "Satellite: custom periods", str(tif.relative_to(LY.ROOT)), "raster",
                         "Sentinel-2 L2A median" if base != "runoff" else "0.4 NDBI + 0.2 low NDVI + 0.2 soil + 0.2 stream proximity",
                         cmap=cmap, vrange=vr, clip=False)
            LY.LAYERS.append(l); LY.BY_ID[lid] = l


# --------------------------------------------------------------------------------------------------------------- exposure
@lru_cache(maxsize=1)
def _zones():
    z = {}
    for lid, T in cfg()["zones"].items():
        g = overview.vector(lid).to_crs(2100)
        z[lid] = (T, g.geometry.make_valid().union_all())
    return z


@lru_cache(maxsize=1)
def _homes():
    bp = pd.read_csv(LY.OUT / "building_population.csv", usecols=["x", "y", "neighbourhood_id", "pop_cal"])
    return gpd.GeoDataFrame(bp, geometry=gpd.points_from_xy(bp.x, bp.y), crs=2100)


def city_exposure(force=False):
    """People and buildings in each hazard zone, city-wide and per neighbourhood. Cached."""
    p = CASE_DIR / "exposure.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    h = _homes()
    total = float(h.pop_cal.sum())
    out, nb_rows = {}, []
    for lid, (T, geom) in _zones().items():
        inside = h.within(geom).to_numpy() if not geom.is_empty else np.zeros(len(h), bool)
        out[lid] = {"return_period_yr": T, "people": round(float(h.pop_cal[inside].sum())), "buildings": int(inside.sum()),
                    "pop_pct": round(100 * float(h.pop_cal[inside].sum()) / total, 3), "area_km2": round(geom.area / 1e6, 3)}
        g = h[inside].groupby("neighbourhood_id").agg(people=("pop_cal", "sum"), buildings=("pop_cal", "size")).reset_index()
        g["zone"] = lid
        nb_rows.append(g)
    nb = pd.concat(nb_rows) if nb_rows else pd.DataFrame()
    names = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")[["neighbourhood_id", "neighbourhood_name"]]
    nb = nb.merge(names, on="neighbourhood_id", how="left")
    nb.round(1).to_csv(CASE_DIR / "exposure_by_neighbourhood.csv", index=False)
    out["total_population"] = round(total)
    p.write_text(json.dumps(out, indent=2))
    return out


def exposure_by_neighbourhood():
    city_exposure()
    return pd.read_csv(CASE_DIR / "exposure_by_neighbourhood.csv")


def exposure_bbox(bbox):
    """Exposure inside a WGS84 bbox (for the API / chat)."""
    from shapely.geometry import box
    area = gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(2100).iloc[0]
    h = _homes()
    h = h[h.within(area)]
    tot = float(h.pop_cal.sum())
    res = {"bbox": list(bbox), "population": round(tot)}
    for lid, (T, geom) in _zones().items():
        inside = h.within(geom).to_numpy()
        res[lid] = {"people": round(float(h.pop_cal[inside].sum())), "buildings": int(inside.sum()),
                    "pop_pct": round(100 * float(h.pop_cal[inside].sum()) / tot, 2) if tot else 0.0}
    return res


# --------------------------------------------------------------------------------------------------------------- network
def network_impact(zone="flood_100", force=False):
    """Walking access to bus stops and parks with and without the street edges inside the flood zone. Cached."""
    p = CASE_DIR / f"access_{zone}.csv"
    if p.exists() and not force:
        return pd.read_csv(p)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    area = overview.municipality()
    streets = gpd.read_file(LY.ROOT / "data/raw/gis/Streets_Center.geojson").to_crs(2100)
    streets = streets[streets.intersects(area)]
    G, _ = network.build_graph(streets, ["raceway", "construction", "proposed", "busway", "elevator"])
    xy = G.graph["xy"]
    keys = list(G.edges())
    geoms = gpd.GeoSeries([G[u][v]["geom"] for u, v in keys], crs=2100)
    geom = _zones()[zone][1]
    flooded = geoms.intersects(geom).to_numpy()
    h = _homes()
    h = h[h.pop_cal > 0]
    node, snap_m, ok = network.snap(G, h[["x", "y"]].to_numpy(), 150)
    h = h[ok].assign(node=node[ok], snap_m=snap_m[ok])
    stops = overview.vector("bus_stops").to_crs(2100)
    sn, _, sok = network.snap(G, np.c_[stops.geometry.x, stops.geometry.y], 150)
    parks = overview.vector("parks").to_crs(2100)
    parks = parks[parks.area >= 5000]
    nodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2100)
    pk = gpd.sjoin(nodes, gpd.GeoDataFrame(geometry=parks.buffer(25), crs=2100), predicate="within").index.unique().to_numpy()
    dest = {"bus_stop": np.unique(sn[sok]), "park": pk}
    in_zone_node = np.zeros(len(xy), bool)
    if not geom.is_empty:
        in_zone_node = gpd.GeoSeries(nodes.geometry).within(geom).to_numpy()
    u = np.array([k[0] for k in keys]); v = np.array([k[1] for k in keys])
    L = np.array([G[a][b]["length"] for a, b in keys])
    rows = []
    for dk, dn in dest.items():
        for state, mask in (("before", np.ones(len(keys), bool)), ("after", ~flooded)):
            dd = dn if state == "before" else dn[~in_zone_node[dn]]
            eu, ev, LL = np.r_[u[mask], v[mask]], np.r_[v[mask], u[mask]], np.r_[L[mask], L[mask]]
            r, _ = abm.walk_to_nearest(len(xy), eu, ev, LL, LL, {}, h.node.to_numpy(), dd)
            d = r.length_m.to_numpy() + h.snap_m.to_numpy()
            df = pd.DataFrame({"nbid": h.neighbourhood_id.to_numpy(), "pop": h.pop_cal.to_numpy(),
                               "w5": np.nan_to_num(d) <= 5 * 60 * SPEED, "reach": np.isfinite(d) & (d <= CUTOFF_M)})
            df.loc[~np.isfinite(d), "w5"] = False
            g = df.assign(w5=df.w5 * df["pop"], reach=df.reach * df["pop"]).groupby("nbid")[["pop", "w5", "reach"]].sum()
            g["dest"], g["state"] = dk, state
            rows.append(g.reset_index())
    t = pd.concat(rows)
    w = t.pivot_table(index=["nbid", "dest"], columns="state", values=["pop", "w5", "reach"], aggfunc="sum")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["within5_pct_before"] = 100 * w.w5_before / w.pop_before
    w["within5_pct_after"] = 100 * w.w5_after / w.pop_before
    w["cut_off_people"] = w.reach_before - w.reach_after
    w["flooded_street_km"] = float(L[flooded].sum() / 1000)
    w = w.rename(columns={"nbid": "neighbourhood_id", "pop_before": "people"})
    w = w[["neighbourhood_id", "dest", "people", "within5_pct_before", "within5_pct_after", "cut_off_people", "flooded_street_km"]]
    w.round(2).to_csv(p, index=False)
    return w


# --------------------------------------------------------------------------------------------------------------- satellite
def _grid(res):
    from rasterio.warp import transform_bounds
    b = transform_bounds("EPSG:4326", "EPSG:32634", *overview.AOI_WGS84)
    w, h = int((b[2] - b[0]) // res), int((b[3] - b[1]) // res)
    return {"crs": "EPSG:32634", "transform": from_origin(b[0], b[3], res, res), "width": w, "height": h}


def spectral(force=False, log=print):
    """The local (default) wet-season indices and runoff susceptibility. Needs network access the first time."""
    c = cfg()["s2"]
    if (CASE_DIR / "runoff.tif").exists() and not force:
        register_layers()
        return json.loads((CASE_DIR / "spectral.json").read_text())
    return _compute(CASE_DIR, c["start"], c["end"], c["max_cloud_pct"], c["max_scenes"], c["resolution_m"], LOCAL_KEY, log=log,
                    extra={})


def spectral_range(start, end, max_cloud_pct=20, max_scenes=6, force=False, log=print, progress=None):
    """Download Sentinel-2 L2A scenes for a user-chosen period from the Planetary Computer STAC and compute the same indices.
    Results go to data/outputs/cases/flood/runs/<YYYYMMDD_YYYYMMDD>/ and appear as layers s2_*__<key>. Cached by period."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if t1 <= t0:
        raise ValueError("end date must be after start date")
    if t0 < pd.Timestamp("2017-01-01"):
        raise ValueError("Sentinel-2 L2A on the Planetary Computer starts in 2017")
    if t0 > pd.Timestamp.now():
        raise ValueError("start date is in the future")
    key = run_key(t0, t1)
    d = run_dir(key)
    if (d / "runoff.tif").exists() and not force:
        register_layers()
        return {"key": key, **json.loads((d / "spectral.json").read_text())}
    d.mkdir(parents=True, exist_ok=True)
    extra = {"start": f"{t0:%Y-%m-%d}", "end": f"{t1:%Y-%m-%d}", "max_cloud_pct": max_cloud_pct, "max_scenes": max_scenes,
             "requested_at": pd.Timestamp.now().isoformat(timespec="seconds")}
    (d / "status.json").write_text(json.dumps({"status": "running", **extra}))
    try:
        meta = _compute(d, f"{t0:%Y-%m-%d}", f"{t1:%Y-%m-%d}", max_cloud_pct, max_scenes, cfg()["s2"]["resolution_m"], key, log, progress,
                        extra)
    except Exception as e:
        (d / "status.json").write_text(json.dumps({"status": "error", "error": str(e), **extra}))
        raise
    (d / "status.json").write_text(json.dumps({"status": "done", **extra}))
    return {"key": key, **meta}


def run_status(key):
    f = run_dir(key) / "status.json"
    return json.loads(f.read_text()) if f.exists() else {"status": "unknown"}


def _compute(out_dir, start, end, max_cloud, max_scenes, res, key, log=print, progress=None, extra=None):
    from pipeline import s2fetch
    prog = progress or (lambda frac, msg: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = _grid(res)
    prog(0.02, "searching the Sentinel-2 catalogue")
    live = {"start": start, "end": end, "max_cloud_pct": max_cloud, "max_scenes": max_scenes}
    items = s2fetch.search_scenes(overview.AOI_WGS84, live)
    if not items:
        raise RuntimeError(f"no Sentinel-2 scenes with < {max_cloud}% cloud between {start} and {end}")
    log(f"  {len(items)} Sentinel-2 scenes: " + ", ".join(i.id for i in items))
    names = ["ndwi", "mndwi", "ndvi", "ndbi"]
    n = len(items)
    done = {"k": 0}

    def scene_log(msg):
        log(msg)
        if "scene" in str(msg):
            done["k"] += 1
            prog(0.05 + 0.8 * done["k"] / n, f"downloading and processing image {done['k']} of {n}")
    comp, nv = s2fetch.composite_indices(items, names, grid, min_valid=1, cache_dir=LY.ROOT / "data" / "cache", log=scene_log)
    prog(0.88, "computing indices and runoff susceptibility")
    prof = dict(driver="GTiff", dtype="float32", count=1, crs=grid["crs"], transform=grid["transform"], width=grid["width"],
                height=grid["height"], nodata=np.nan, compress="deflate")
    for nm in names:
        with rasterio.open(out_dir / f"s2_{nm}.tif", "w", **prof) as dst:
            dst.write(comp[nm].astype("float32"), 1)
    shp = (grid["height"], grid["width"])
    soils = overview.vector("soil_groups").to_crs(grid["crs"])
    soil = rasterize(((g, SOIL_RUNOFF.get(s_, 0.5)) for g, s_ in zip(soils.geometry, soils.SoilGroup)), out_shape=shp,
                     transform=grid["transform"], fill=0.5, dtype="float32")
    rivers = overview.vector("rivers").to_crs(grid["crs"])
    rv = rasterize(((g, 1) for g in rivers.geometry), out_shape=shp, transform=grid["transform"], fill=0, dtype="uint8")
    dist = distance_transform_edt(rv == 0) * res if rv.any() else np.full(shp, 5000.0)
    near = np.exp(-dist / 300.0)

    def nrm(a, lo, hi):
        return np.clip((a - lo) / (hi - lo), 0, 1)
    runoff = (0.4 * nrm(comp["ndbi"], -0.3, 0.2) + 0.2 * (1 - nrm(comp["ndvi"], 0.1, 0.6)) + 0.2 * soil + 0.2 * near).astype("float32")
    runoff[~np.isfinite(comp["ndbi"])] = np.nan
    with rasterio.open(out_dir / "runoff.tif", "w", **prof) as dst:
        dst.write(runoff, 1)
    nb = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")[["neighbourhood_id", "neighbourhood_name", "geometry"]].to_crs(grid["crs"])
    lab = rasterize(((g, i + 1) for i, g in enumerate(nb.geometry)), out_shape=shp, transform=grid["transform"], fill=0, dtype="int32")
    sr = pd.Series(runoff.ravel()).groupby(lab.ravel()).mean()
    nb["runoff_mean"] = [float(sr.get(i + 1, np.nan)) for i in range(len(nb))]
    nb.drop(columns="geometry").round(3).to_csv(out_dir / "runoff_by_neighbourhood.csv", index=False)
    meta = {"scenes": s2fetch.scene_table(items), "grid_res_m": res, "weights": {"ndbi": 0.4, "low_ndvi": 0.2, "soil": 0.2, "stream": 0.2},
            "status": "done", **(extra or {})}
    (out_dir / "spectral.json").write_text(json.dumps(meta, indent=2))
    prog(0.93, "rendering map layers")
    register_layers()
    for base in SAT_LAYERS:
        overview.prepare_layer(layer_id(base, key), force=True, log=log)
    prog(1.0, "done")
    return meta


def runoff_by_neighbourhood(key=LOCAL_KEY):
    p = run_dir(key) / "runoff_by_neighbourhood.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


register_layers()
