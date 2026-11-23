"""City of Athens Smart-Neighbourhood environmental stations (opendata.cityofathens.gr): 100 stations with hourly air temperature,
humidity, pressure, AQI, PM1/PM2.5/PM10, CO, NO2, O3, SO2.

The station registry has addresses but no coordinates, so stations are geocoded once (OpenStreetMap Nominatim, 1 request/s,
cached in data/raw/environment/stations.geojson). Where an address does not resolve, the station is placed at its neighbourhood's
centre and flagged `approximate`. Readings come from the aggregate hourly dataset already used by the pipeline
(data/raw/environment/aggregate.parquet, refreshed by make_crosswalk.py); timestamps are UTC.

The map layer shows, per station: the latest reading, the summer-afternoon mean air temperature (Jul–Aug, 12:00–17:00 local),
the neighbourhood's satellite surface temperature for comparison, and a 7-day temperature sparkline."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import layers as LY

ENV = LY.ROOT / "data" / "raw" / "environment"
REGISTRY_URL = "https://opendata.cityofathens.gr/api/3/action/datastore_search?resource_id=47ba486c-4ecf-497e-892b-4e47c53f422a&limit=500"
REG = ENV / "stations_registry.json"
GEO = ENV / "stations_geocoded.json"
LAYER = ENV / "stations.geojson"
UA = "athens-urban-intelligence/1.0 (open-data research)"
ATHENS_VIEWBOX = "23.68,38.05,23.80,37.93"


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def registry(refresh=False) -> pd.DataFrame:
    if not REG.exists() or refresh:
        ENV.mkdir(parents=True, exist_ok=True)
        REG.write_bytes(_get(REGISTRY_URL))
    return pd.DataFrame(json.loads(REG.read_text())["result"]["records"])


def geocode(refresh=False, log=print) -> dict:
    """{station_id: (lon, lat, 'address'|'neighbourhood')} via Nominatim, cached. Falls back to the neighbourhood centre."""
    cache = json.loads(GEO.read_text()) if GEO.exists() and not refresh else {}
    reg = registry()
    nb = gpd.read_file(LY.ROOT / "data/raw/gis/athens_neighbourhoods.geojson").to_crs(2100)
    nb_pt = nb.set_index(nb["id"].astype(int)).geometry.representative_point().to_crs(4326)
    changed = False
    for r in reg.itertuples():
        k = str(r.station_id)
        if k in cache:
            continue
        hit = None
        for q in (f"{r.station_address}, Αθήνα", f"{r.station_name}, Αθήνα"):
            if not isinstance(q, str) or q.startswith("nan"):
                continue
            url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
                {"q": q, "format": "json", "limit": 1, "viewbox": ATHENS_VIEWBOX, "bounded": 1, "countrycodes": "gr"})
            try:
                res = json.loads(_get(url, 30))
            except Exception:
                res = []
            time.sleep(1.1)                               # Nominatim usage policy: max 1 request per second
            if res:
                hit = (float(res[0]["lon"]), float(res[0]["lat"]), "address")
                break
        if hit is None and int(r.neighbourhood_id) in nb_pt.index:
            p = nb_pt.loc[int(r.neighbourhood_id)]
            hit = (float(p.x), float(p.y), "neighbourhood")
        if hit:
            cache[k] = hit
            changed = True
            log(f"  station {k}: {hit[2]}")
    if changed:
        GEO.write_text(json.dumps(cache, indent=1))
    return cache


def _readings():
    d = pd.read_parquet(ENV / "aggregate.parquet")
    d["t"] = pd.to_datetime(d.interval_timestamp)                   # UTC
    d["t_local"] = d.t.dt.tz_localize("UTC").dt.tz_convert("Europe/Athens").dt.tz_localize(None)
    return d


def build_layer(log=print) -> gpd.GeoDataFrame:
    """Station points with latest values, summer-afternoon air temperature, neighbourhood LST and a 7-day sparkline."""
    reg = registry()
    geo = geocode(log=log)
    d = _readings()
    ind = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")
    lst = dict(zip(ind.neighbourhood_id, ind.lst_p90_c))
    end = d.t.max()
    summer = d[(d.t_local.dt.month.isin([7, 8])) & d.t_local.dt.hour.between(12, 17)]
    rows = []
    for r in reg.itertuples():
        g = geo.get(str(r.station_id))
        s = d[d.station_id == r.station_id].sort_values("t")
        if g is None or s.empty:
            continue
        last = s.dropna(subset=["temperature"]).iloc[-1] if s.temperature.notna().any() else s.iloc[-1]
        wk = s[s.t > end - pd.Timedelta(days=7)].set_index("t").temperature.resample("3h").mean()
        sm = summer[summer.station_id == r.station_id].temperature
        nbid = f"athens:nb:{int(r.neighbourhood_id):03d}"
        rows.append({
            "station_id": int(r.station_id), "name": r.station_name, "address": r.station_address, "neighbourhood": r.neighbourhood_name,
            "district": int(r.municipal_district_id), "location": "approximate (neighbourhood centre)" if g[2] == "neighbourhood" else "address",
            "last_time_local": str(last.t_local)[:16], "temp_c": _r(last.temperature), "humidity_pct": _r(last.humidity),
            "aqi": _r(last.aqi, 0), "pm2_5": _r(last.pm2_5), "no2": _r(last.no2), "o3": _r(last.o3),
            "summer_pm_temp_c": _r(sm.mean()) if len(sm) else None, "summer_pm_temp_max_c": _r(sm.max()) if len(sm) else None,
            "nb_lst_p90_c": _r(lst.get(nbid)), "spark": [None if not np.isfinite(v) else round(float(v), 1) for v in wk.values],
            "spark_from": str(wk.index.min())[:10] if len(wk) else "", "geometry": gpd.points_from_xy([g[0]], [g[1]])[0]})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    gdf.to_file(LAYER, driver="GeoJSON")
    return gdf


def _r(v, nd=1):
    try:
        v = float(v)
        return None if not np.isfinite(v) else round(v, nd)
    except Exception:
        return None


def station_series(station_id, days=14):
    """Hourly series (local time) for one station, for the API."""
    d = _readings()
    s = d[d.station_id == int(station_id)].sort_values("t")
    s = s[s.t > s.t.max() - pd.Timedelta(days=days)]
    return s[["t_local", "temperature", "humidity", "aqi", "pm2_5", "pm10", "no2", "o3"]].rename(columns={"t_local": "time"})
