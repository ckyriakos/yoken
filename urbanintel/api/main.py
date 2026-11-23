"""Athens Urban Intelligence API (v1).

Run from the project root:   uvicorn urbanintel.api.main:app --port 8000      (docs at /docs)
Everything here calls urbanintel.services, the same code the demo UI uses. The endpoints are also the tool surface for the planned
natural-language assistant: each one is a small, typed, side-effect-free query (simulations are cached by parameters)."""
from __future__ import annotations

import json
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .. import layers as LY
from ..services import flood, heat, landcover, measures, mobility, overview, sensors, timelapse, traffic

app = FastAPI(title="Athens Urban Intelligence API", version="1.0.0",
              description="Open map layers, neighbourhood indicators and agent-based what-if simulations for the City of Athens.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
V1 = "/api/v1"


def _bbox(bbox: str | None):
    if not bbox:
        return None
    try:
        b = tuple(float(x) for x in bbox.split(","))
        assert len(b) == 4 and b[0] < b[2] and b[1] < b[3]
        return b
    except Exception:
        raise HTTPException(400, "bbox must be 'minLon,minLat,maxLon,maxLat'")


def _json(obj):
    """NaN-safe JSON response."""
    return JSONResponse(json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("NaN", "null")))


def _df(df: pd.DataFrame):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records", force_ascii=False))


# ------------------------------------------------------------------------------------------------------------ meta
@app.get("/", include_in_schema=False)
def root():
    return {"name": app.title, "docs": "/docs", "api": V1}


@app.get(f"{V1}/health")
def health():
    return {"status": "ok", "layers_prepared": sum(LY.prepared_path(l).exists() for l in LY.LAYERS), "layers": len(LY.LAYERS)}


# ------------------------------------------------------------------------------------------------------------ layers
@app.get(f"{V1}/layers", summary="Layer catalogue")
def list_layers(group: str | None = None):
    flood.register_layers()
    return [l for l in overview.catalogue() if group is None or l["group"] == group]


@app.get(f"{V1}/layers/{{layer_id}}", summary="Vector layer as GeoJSON (EPSG:4326)")
def get_layer(layer_id: str, bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat"),
              simplify: float | None = Query(None, description="tolerance in degrees"), limit: int | None = None):
    try:
        layer = LY.get(layer_id)
        if layer.kind == "raster":
            return {"id": layer_id, "kind": "raster", **overview.raster_meta(layer_id),
                    "png": f"{V1}/layers/{layer_id}/image.png", "tiles": f"{V1}/layers/{layer_id}/tiles/{{z}}/{{x}}/{{y}}.png"}
        return _json(overview.geojson(layer_id, bbox=_bbox(bbox), simplify=simplify, limit=limit))
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get(f"{V1}/layers/{{layer_id}}/image.png", summary="Raster layer as a single coloured PNG overlay (see bounds in /layers/{id})")
def layer_png(layer_id: str):
    try:
        return Response(overview.raster_png(layer_id), media_type="image/png")
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get(f"{V1}/layers/{{layer_id}}/tiles/{{z}}/{{x}}/{{y}}.png", summary="Raster layer as XYZ tiles")
def layer_tile(layer_id: str, z: int, x: int, y: int):
    try:
        b = overview.tile(layer_id, z, x, y)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return Response(b or b"", status_code=200 if b else 204, media_type="image/png")


# ------------------------------------------------------------------------------------------------------------ indicators
@app.get(f"{V1}/indicators/overview", summary="City headline numbers")
def indicators_overview(districts: str | None = Query(None, description="comma-separated district numbers 1-7; default all")):
    return _json(overview.headline(_dks(districts)))


@app.get(f"{V1}/districts", summary="The 7 municipal districts (Δημοτικές Κοινότητες)")
def list_districts():
    return [{"id": k, "label": v} for k, v in overview.districts().items()]


def _dks(districts):
    if not districts:
        return ()
    try:
        return overview.norm_districts([int(x) for x in districts.split(",") if x.strip()])
    except ValueError:
        raise HTTPException(400, "districts must be numbers 1-7, e.g. 2,4")


@app.get(f"{V1}/indicators/neighbourhoods", summary="Heat, greenery, access and flood indicators per neighbourhood")
def indicators_neighbourhoods(format: Literal["json", "geojson"] = "json", districts: str | None = Query(None, description="e.g. 2,4")):
    g = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")
    if districts:
        g = g[g.neighbourhood_id.isin(overview.nb_ids(_dks(districts)))]
    cols = ["neighbourhood_id", "neighbourhood_name", "neighbourhood_name_en", "lst_p90_c", "ndvi_mean", "priority_score", "priority_rank",
            "hotspot_share_pct", "geometry"]
    g = g[cols]
    try:
        ac = pd.read_csv(LY.OUT / "neighbourhood_access.csv")[["neighbourhood_id", "population", "green_open_within5_pct", "bus_stop_within5_pct"]]
        g = g.merge(ac, on="neighbourhood_id", how="left")
    except FileNotFoundError:
        pass
    ex = flood.exposure_by_neighbourhood()
    if len(ex):
        piv = ex.pivot_table(index="neighbourhood_id", columns="zone", values="people", aggfunc="sum").add_prefix("people_in_").reset_index()
        g = g.merge(piv, on="neighbourhood_id", how="left")
    if format == "geojson":
        return _json(json.loads(g.to_json(drop_id=True)))
    return _df(pd.DataFrame(g.drop(columns="geometry")))


# ------------------------------------------------------------------------------------------------------------ cases
CASES = {"kallimarmaro": "Vas. Olgas redevelopment 2026: traffic ABM + Sentinel-2 land-cover change", "heat": "Heat proposals simulated (heat ABM)",
         "flood": "Flood exposure (zones, network, Sentinel-2)"}


@app.get(f"{V1}/cases", summary="Available case studies")
def cases():
    return [{"id": k, "title": v, "url": f"{V1}/cases/{k}"} for k, v in CASES.items()]


@app.get(f"{V1}/cases/{{case_id}}", summary="Cached case-study results")
def case(case_id: str):
    if case_id == "kallimarmaro":
        out = dict(mobility.result())
        p = landcover.OUT / "landcover.json"
        if p.exists():
            out["landcover"] = json.loads(p.read_text())
        return _json(out)
    if case_id == "heat":
        return _json(heat.run_case())
    if case_id == "flood":
        out = {"exposure": flood.city_exposure(), "by_neighbourhood": _df(flood.exposure_by_neighbourhood())}
        try:
            out["network_flood_100"] = _df(flood.network_impact())
        except Exception as e:      # noqa: BLE001
            out["network_flood_100"] = {"error": str(e)}
        out["runoff_by_neighbourhood"] = _df(flood.runoff_by_neighbourhood())
        return _json(out)
    raise HTTPException(404, f"unknown case '{case_id}'. Known: {', '.join(CASES)}")


@app.get(f"{V1}/cases/kallimarmaro/layers/{{name}}", summary="Vas. Olgas map layers: edges (car volume, shade per scenario), traces, olgas, signals")
def kallimarmaro_layer(name: Literal["edges", "traces", "olgas", "signals"]):
    return _json(mobility.layer(name))


@app.get(f"{V1}/cases/kallimarmaro/landcover", summary="Sentinel-2 NDVI / land-cover change on Vas. Olgas: before works, during, after")
def kallimarmaro_landcover(refresh: bool = False):
    try:
        r = landcover.run(force=refresh)
    except Exception as e:      # noqa: BLE001
        raise HTTPException(503, f"Sentinel-2 fetch failed: {e}")
    return _json({**r, "layers": {k: f"{V1}/layers/{k}" for k in landcover.LAYER_IDS}, "zones": landcover.zones()})


@app.get(f"{V1}/cases/kallimarmaro/neighbourhoods", summary="Pedestrian metrics per neighbourhood and scenario")
def kallimarmaro_nb():
    return _df(mobility.by_neighbourhood())


class HeatParams(BaseModel):
    tree_budget_m: float = Field(1000, ge=0, le=10000, description="metres of street to plant")
    refuges: int = Field(0, ge=0, le=12, description="number of cooling refuges to open")
    neighbourhoods: list[str] | None = Field(None, description="neighbourhood ids, e.g. athens:nb:033; default = top priority")


@app.post(f"{V1}/simulate/heat", summary="Run (or fetch cached) heat ABM for a proposal")
def simulate_heat(p: HeatParams, include_layers: bool = False):
    r = heat.cached(p.tree_budget_m, p.refuges, p.neighbourhoods)
    if not include_layers:
        r = {k: v for k, v in r.items() if k != "layers"}
    return _json(r)


class KallimarmaroParams(BaseModel):
    force: bool = Field(False, description="re-run instead of returning the cached result (about 30 s)")
    seeds: int | None = Field(None, ge=1, le=20)


@app.post(f"{V1}/simulate/kallimarmaro", summary="Run (or fetch cached) Vas. Olgas mobility ABM, all scenarios")
def simulate_kallimarmaro(p: KallimarmaroParams):
    return _json(mobility.run_case(force=p.force, seeds=p.seeds))


# ------------------------------------------------------------------------------------------------------------ sensors, measures
@app.get(f"{V1}/sensors/environment", summary="City of Athens environmental stations: latest reading, summer-afternoon air temp.")
def env_stations():
    p = sensors.LAYER
    if not p.exists():
        sensors.build_layer(log=lambda *_: None)
    return _json(json.loads(p.read_text()))


@app.get(f"{V1}/sensors/environment/{{station_id}}/series", summary="Hourly readings of one station (local time)")
def env_series(station_id: int, days: int = Query(14, ge=1, le=120)):
    d = sensors.station_series(station_id, days)
    if d.empty:
        raise HTTPException(404, f"no data for station {station_id}")
    return _df(d.assign(time=d.time.astype(str)))


@app.get(f"{V1}/neighbourhoods/{{neighbourhood_id}}/measures", summary="Area type and cooling measures that fit a neighbourhood")
def nb_measures(neighbourhood_id: str):
    r = measures.for_neighbourhoods([neighbourhood_id])
    if not r:
        raise HTTPException(404, f"unknown neighbourhood '{neighbourhood_id}' (ids look like athens:nb:033)")
    return _json(r[0])


# ------------------------------------------------------------------------------------------------------------ traffic
_LIVE = {"t": 0.0, "data": None}


@app.get(f"{V1}/traffic/live", summary="Latest Region of Attica loop-detector readings (veh/h, speed); cached 60 s")
def traffic_live(bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat")):
    import time
    if _LIVE["data"] is None or time.time() - _LIVE["t"] > 60:
        try:
            _LIVE["data"], _LIVE["t"] = traffic.live(), time.time()
        except Exception as e:      # noqa: BLE001
            raise HTTPException(503, f"Attica traffic feed unavailable: {e}")
    d = _LIVE["data"]
    d = d[d.vehicle == "all"]
    b = _bbox(bbox)
    if b:
        d = d[d.longitude.between(b[0], b[2]) & d.latitude.between(b[1], b[3])]
    cols = ["id", "timestamp", "count", "occ", "speed", "odos", "perigrafi_thesis", "latitude", "longitude"]
    return _df(d[cols].assign(timestamp=d.timestamp.astype(str)).rename(columns={"count": "vph", "perigrafi_thesis": "description"}))


@app.get(f"{V1}/traffic/history", summary="Logged detector readings, hourly mean veh/h (needs `python -m urbanintel.collect_traffic` running)")
def traffic_history(ids: str = Query(..., description="comma-separated detector ids, e.g. MS358,MS231"), start: str | None = None,
                    end: str | None = None):
    h = traffic.hourly(traffic.logged(ids=ids.split(","), start=start, end=end))
    return _df(h.assign(hour=h.hour.astype(str)))


@app.get(f"{V1}/cases/kallimarmaro/calibration", summary="Vas. Olgas car model vs Attica loop detectors")
def kallimarmaro_calibration():
    r = traffic.calibration_result()
    if r is None:
        raise HTTPException(404, "not calibrated yet: run `python -m urbanintel.calibrate`")
    return _json(r)


# ------------------------------------------------------------------------------------------------------------ flood
@app.get(f"{V1}/flood/exposure", summary="People and buildings in flood zones (city or bbox)")
def flood_exposure(bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat")):
    b = _bbox(bbox)
    return _json(flood.exposure_bbox(b) if b else flood.city_exposure())


@app.get(f"{V1}/flood/network", summary="Walking access lost when 100-year flood streets are closed")
def flood_network():
    return _df(flood.network_impact())


class IndexRequest(BaseModel):
    start: str | None = Field(None, description="YYYY-MM-DD; omit start and end for the local wet-season set (default)")
    end: str | None = Field(None, description="YYYY-MM-DD")
    max_cloud_pct: int = Field(20, ge=1, le=100)
    max_scenes: int = Field(6, ge=1, le=12)
    refresh: bool = Field(False, description="recompute even if this period is cached")


_RUNS: dict = {}


@app.post(f"{V1}/indices", summary="Sentinel-2 NDWI / MNDWI / NDVI / NDBI + runoff susceptibility for the local set or any period")
def indices(r: IndexRequest):
    """Without dates: the local wet-season set (computed once, then cached).
    With start/end: downloads Sentinel-2 L2A scenes for that period from the Planetary Computer and computes the indices in the
    background. Returns immediately with status 'running'; poll GET /indices/periods/{key}."""
    import threading
    if not r.start and not r.end:
        try:
            meta = flood.spectral(force=r.refresh)
        except Exception as e:      # noqa: BLE001
            raise HTTPException(503, f"Sentinel-2 fetch failed: {e}")
        return {"key": flood.LOCAL_KEY, "status": "done", **meta, "layers": {k: f"{V1}/layers/{k}" for k in flood.SAT_LAYERS}}
    if not (r.start and r.end):
        raise HTTPException(400, "give both start and end, or neither")
    try:
        key = flood.run_key(r.start, r.end)
    except Exception:
        raise HTTPException(400, "dates must be YYYY-MM-DD")
    st = flood.run_status(key)
    if st.get("status") == "done" and not r.refresh:
        return _period(key)
    t = _RUNS.get(key)
    if t is None or not t.is_alive():
        def go():
            try:
                flood.spectral_range(r.start, r.end, r.max_cloud_pct, r.max_scenes, force=r.refresh)
            except Exception:
                pass                                    # the error is recorded in status.json
        try:
            import pandas as pd_
            if pd_.Timestamp(r.end) <= pd_.Timestamp(r.start):
                raise ValueError("end date must be after start date")
        except ValueError as e:
            raise HTTPException(400, str(e))
        _RUNS[key] = threading.Thread(target=go, daemon=True)
        _RUNS[key].start()
    return {"key": key, "status": "running", "poll": f"{V1}/indices/periods/{key}"}


def _period(key):
    flood.register_layers()
    p = next((x for x in flood.periods() if x["key"] == key), None)
    if p is None:
        st = flood.run_status(key)
        if st.get("status") == "unknown":
            raise HTTPException(404, f"no period '{key}'")
        return {"key": key, **st}
    return {**p, "layers": {b: f"{V1}/layers/{flood.layer_id(b, key)}" for b in flood.SAT_LAYERS}}


@app.get(f"{V1}/indices/periods", summary="Available Sentinel-2 periods (local default first)")
def indices_periods():
    return [{k: v for k, v in p.items() if k != "scenes"} | {"n_scenes": len(p.get("scenes", []))} for p in flood.periods()]


@app.get(f"{V1}/indices/periods/{{key}}", summary="One Sentinel-2 period: status, scenes and layer URLs")
def indices_period(key: str):
    return _json(_period(key))


# ------------------------------------------------------------------------------------------------------------ timelapse
class TimelapseRequest(BaseModel):
    sensor: Literal["sentinel-2", "landsat"] = "sentinel-2"
    product: Literal["true_color", "ndvi", "ndwi", "mndwi", "ndbi", "lst"] = "ndvi"
    start_year: int = Field(2018, ge=1984)
    end_year: int = Field(2026, ge=1984)
    season: str = Field("Summer (Jun–Sep)", description=f"one of {list(timelapse.SEASONS)}")
    step: Literal["yearly", "monthly"] = "yearly"
    districts: list[int] = Field(default_factory=list, description="municipal districts 1-7; empty = whole municipality")
    max_cloud_pct: int = Field(20, ge=1, le=100)
    max_scenes: int = Field(3, ge=1, le=8)
    frame_ms: int = Field(900, ge=100, le=5000)


_TL: dict = {}


@app.post(f"{V1}/timelapse", summary="Build a satellite timelapse (downloads scenes per time step, in the background)")
def timelapse_build(r: TimelapseRequest):
    """Returns the timelapse key immediately; poll GET /timelapse/{key}. Cached by parameters."""
    import threading
    if r.season not in timelapse.SEASONS:
        raise HTTPException(400, f"season must be one of {list(timelapse.SEASONS)}")
    if r.product == "lst" and r.sensor != "landsat":
        raise HTTPException(400, "surface temperature (lst) needs sensor=landsat")
    try:
        timelapse.windows(max(r.start_year, timelapse.SENSORS[r.sensor]["first_year"]), r.end_year, timelapse.SEASONS[r.season], r.step)
    except ValueError as e:
        raise HTTPException(400, str(e))
    dks = overview.norm_districts(r.districts)
    params = dict(sensor=r.sensor, product=r.product, start_year=max(r.start_year, timelapse.SENSORS[r.sensor]["first_year"]),
                  end_year=r.end_year, months=list(timelapse.SEASONS[r.season]), step=r.step, districts=list(dks),
                  max_cloud_pct=r.max_cloud_pct, max_scenes=r.max_scenes)
    k = timelapse.key(**params)
    m = timelapse.get(k)
    if m and m.get("status") == "done":
        return _tl(m)
    t = _TL.get(k)
    if t is None or not t.is_alive():
        def go():
            try:
                timelapse.build(r.sensor, r.product, r.start_year, r.end_year, r.season, r.step, dks, r.max_cloud_pct, r.max_scenes, r.frame_ms)
            except Exception:
                pass                                     # recorded in meta.json
        _TL[k] = threading.Thread(target=go, daemon=True)
        _TL[k].start()
    return {"key": k, "status": "running", "poll": f"{V1}/timelapse/{k}"}


def _tl(m):
    k = m["key"]
    out = {kk: v for kk, v in m.items() if kk != "frames"}
    out["frames"] = [{**f, "frame_url": f"{V1}/timelapse/{k}/{f['frame']}", "overlay_url": f"{V1}/timelapse/{k}/{f['overlay']}"}
                     for f in m.get("frames", [])]
    if m.get("status") == "done":
        out["gif_url"] = f"{V1}/timelapse/{k}/timelapse.gif"
    return out


@app.get(f"{V1}/timelapse", summary="Saved timelapses")
def timelapse_list():
    return [{k: v for k, v in m.items() if k not in ("frames",)} | {"n_frames": len(m.get("frames", []))} for m in timelapse.listing()]


@app.get(f"{V1}/timelapse/{{key}}", summary="Timelapse status, frames (dates, area mean) and file URLs")
def timelapse_get(key: str):
    m = timelapse.get(key)
    if m is None:
        raise HTTPException(404, f"no timelapse '{key}'")
    return _json(_tl(m))


@app.get(f"{V1}/timelapse/{{key}}/{{rel:path}}", summary="Timelapse files: timelapse.gif, frames/NN.png, map/NN.png (overlay)")
def timelapse_file(key: str, rel: str):
    try:
        p = timelapse.path(key, rel)
    except ValueError:
        raise HTTPException(400, "bad path")
    if not p.exists() or p.suffix not in (".gif", ".png"):
        raise HTTPException(404, "not found")
    return FileResponse(p, media_type="image/gif" if p.suffix == ".gif" else "image/png")
