"""Region of Attica loop-detector counts (data.gov.gr, CC BY 4.0) and calibration of the Vas. Olgas car model against them.

Sources
  live:     https://loader.data.gov.gr/data/attica-region-traffic-data  (newest snapshot only, no key, <= 60 requests/min)
            ~620 detectors; `count` is vehicles/hour (a 90 s sample x 40), plus occupancy %, speed km/h, street and WGS84 position.
  history:  data.gov.gr dataset road_traffic_attica export: hourly rows for Sat 22 - Sun 23 Nov 2025 (Vas. Olgas still closed).
            `countedcars` is the SUM of the ~40 x 90 s veh/h samples in the hour, so veh/h = countedcars / 40 (inferred: it matches the
            live readings at the same stations; not documented by the publisher).
The live feed keeps no history, so `collect()` logs it to data/raw/traffic/attica/YYYY-MM-DD.csv.gz (python -m urbanintel.collect_traffic).
"""
from __future__ import annotations

import gzip
import io
import json
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .. import layers as LY

LIVE_URL = "https://loader.data.gov.gr/data/attica-region-traffic-data"
HIST_PAGE = "https://data.gov.gr/dataset/road_traffic_attica"
HIST_URL = ("https://data.gov.gr/dataset/1d164912-9dd7-421b-b6f8-5349f9f15d0a/resource/ca9fdbca-4bdf-450c-90fa-a6d5a9081344/download/"
            "export_road_traffic_attica.csv")
UA = "athens-urban-intelligence/1.0 (research; open data)"
RAW = LY.ROOT / "data" / "raw" / "traffic"
LOG_DIR = RAW / "attica"
STATIONS = RAW / "stations.geojson"
HIST = RAW / "history_2025-11-22_23.csv"
HIST_SAMPLES_PER_HOUR = 40
TZ = "Europe/Athens"


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def norm(s):
    """Upper-case Greek without accents, for street-name matching."""
    s = unicodedata.normalize("NFD", str(s or "")).upper()
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def street_key(odos):
    """Main word of a detector street name: 'Λ. ΑΜΑΛΙΑΣ' -> 'ΑΜΑΛΙΑΣ', 'ΒΑΣ. ΚΩΝΣΤΑΝΤΙΝΟΥ' -> 'ΚΩΝΣΤΑΝΤΙΝΟΥ'."""
    words = [w for w in norm(odos).replace("(", " ").replace(")", " ").split() if not w.endswith(".") and len(w) > 2]
    return max(words, key=len) if words else ""


# ------------------------------------------------------------------------------------------------------------ live + log
def live() -> pd.DataFrame:
    """Newest snapshot of every detector (one row per detector and vehicle class)."""
    df = pd.read_json(io.BytesIO(_get(LIVE_URL)), lines=True)
    for c in ("count", "occ", "speed"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    return df


def stations(refresh=False) -> gpd.GeoDataFrame:
    """Detector metadata (id, street, description, position). Cached from the live feed."""
    if STATIONS.exists() and not refresh:
        return gpd.read_file(STATIONS)
    d = live()
    d = d[d.vehicle == "all"].drop_duplicates("id")
    g = gpd.GeoDataFrame(d[["id", "odos", "perigrafi_thesis", "dimos", "anichneytes", "count", "speed", "timestamp"]]
                         .rename(columns={"perigrafi_thesis": "description", "anichneytes": "loops", "count": "last_vph",
                                          "speed": "last_speed", "timestamp": "last_time"}),
                         geometry=gpd.points_from_xy(d.longitude, d.latitude), crs=4326)
    g["last_time"] = g.last_time.astype(str)
    RAW.mkdir(parents=True, exist_ok=True)
    g.to_file(STATIONS, driver="GeoJSON")
    return g


def collect(state=None):
    """One poll: append readings not seen before to today's log. Returns number of new rows."""
    d = live()
    d = d[d.vehicle == "all"]
    state = {} if state is None else state
    new = d[[state.get(i) != t for i, t in zip(d.id, d.timestamp.astype(str))]]
    for i, t in zip(new.id, new.timestamp.astype(str)):
        state[i] = t
    if len(new):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = LOG_DIR / f"{day}.csv.gz"
        head = not p.exists()
        with gzip.open(p, "at") as f:
            new[["id", "timestamp", "count", "occ", "speed"]].to_csv(f, header=head, index=False)
    return len(new)


def logged(ids=None, start=None, end=None) -> pd.DataFrame:
    """Logged live readings (UTC timestamps), optionally filtered."""
    files = sorted(LOG_DIR.glob("*.csv.gz"))
    if not files:
        return pd.DataFrame(columns=["id", "timestamp", "count", "occ", "speed"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, format="mixed")
    if ids is not None:
        df = df[df.id.isin(list(ids))]
    if start is not None:
        df = df[df.timestamp >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.timestamp <= pd.Timestamp(end, tz="UTC")]
    return df.drop_duplicates(["id", "timestamp"])


def history(refresh=False) -> pd.DataFrame:
    """Hourly veh/h for Sat 22 - Sun 23 Nov 2025 (Vas. Olgas closed). Columns id, hour (local), vph, speed."""
    if not HIST.exists() or refresh:
        RAW.mkdir(parents=True, exist_ok=True)
        HIST.write_bytes(_get(HIST_URL, timeout=180))
    h = pd.read_csv(HIST)
    return pd.DataFrame({"id": h.deviceid, "hour": pd.to_datetime(h.appprocesstime), "vph": h.countedcars / HIST_SAMPLES_PER_HOUR,
                         "speed": h.average_speed, "odos": h.road_name})


def hourly(df) -> pd.DataFrame:
    """Logged readings -> mean veh/h per detector and local hour."""
    if df.empty:
        return pd.DataFrame(columns=["id", "hour", "vph", "n"])
    t = df.timestamp.dt.tz_convert(TZ).dt.tz_localize(None).dt.floor("h")
    return df.assign(hour=t).groupby(["id", "hour"]).agg(vph=("count", "mean"), n=("count", "size")).reset_index()


# ------------------------------------------------------------------------------------------------------------ calibration
def match_edges(st: gpd.GeoDataFrame, e: gpd.GeoDataFrame, max_m=60):
    """Each detector -> drivable model edges of the same street within max_m (both carriageways). Returns {id: [edge index]}."""
    s = st.to_crs(2100)
    ee = e[e.drivable].to_crs(2100)
    names = ee.name_s.map(norm)
    out = {}
    for sid, odos, p in zip(s.id, s.odos, s.geometry):
        k = street_key(odos)
        if not k:
            continue
        cand = ee[names.str.contains(k, regex=False).to_numpy() & (ee.distance(p) <= max_m).to_numpy()]
        if len(cand):
            out[sid] = cand.index.tolist()
    return out


def calibration(observed: pd.DataFrame, model_vol: np.ndarray, matches: dict, label: str) -> dict:
    """Compare observed veh/h (DataFrame id, vph) with model volumes on matched edges.
    A detector covers one direction; the model value is the MEAN of its matched edges (one per carriageway), i.e. a typical direction.
    Returns per-detector table and the demand scale factor (median observed / model)."""
    rows = []
    for sid, vph in zip(observed.id, observed.vph):
        if sid in matches and np.isfinite(vph) and vph > 0:
            m = float(np.mean(model_vol[matches[sid]]))
            rows.append({"id": sid, "observed_vph": round(float(vph)), "model_vph": round(m), "ratio": round(vph / m, 2) if m > 0 else None})
    t = pd.DataFrame(rows)
    ok = t[t.model_vph > 0] if len(t) else t
    return {"label": label, "n_detectors": int(len(t)),
            "scale_median": round(float(ok.ratio.median()), 2) if len(ok) else None,
            "scale_iqr": [round(float(ok.ratio.quantile(.25)), 2), round(float(ok.ratio.quantile(.75)), 2)] if len(ok) else None,
            "mape_pct": round(float((abs(ok.model_vph - ok.observed_vph) / ok.observed_vph).mean() * 100), 1) if len(ok) else None,
            "detectors": t.to_dict("records")}


def near_olgas(radius_m=1200, main_road_only=False):
    """Detectors within radius_m of Vas. Olgas (optionally only main-carriageway detectors)."""
    from . import mobility
    st = stations()
    if main_road_only:
        ex = mobility.cfg().get("calibration", {}).get("exclude_desc", [])
        st = st[~st.description.astype(str).map(norm).apply(lambda d: any(norm(x) in d for x in ex))]
    olg = gpd.read_file(mobility.CASE_DIR / "olgas.geojson").to_crs(2100).geometry.union_all()
    d = st.to_crs(2100).distance(olg)
    return st[(d <= radius_m).to_numpy()].assign(dist_olgas_m=d[d <= radius_m].round(0).to_numpy())


def run_calibration(log=print):
    """Model (seed-0 edge volumes from the cached run) vs detectors:
      before: Nov 2025 weekend history (Olgas closed) vs model 'baseline', at the busiest Saturday daytime hour;
      after:  logged live weekday peak (07-10 and 17-20 local) vs model 'as_built', once the logger has collected weekdays.
    Writes data/outputs/cases/kallimarmaro/calibration.json."""
    from . import mobility
    nodes, e = mobility.network()
    full = _seed0_volumes()
    st = near_olgas(mobility.cfg().get("calibration", {}).get("max_dist_m", 800), main_road_only=True)
    matches = match_edges(st, e)
    out = {"detectors_near_olgas": int(len(st)), "matched": int(len(matches)), "sources": {"live": LIVE_URL, "history": HIST_PAGE},
           "demand_scale": mobility.cfg()["cars"].get("demand_scale", 1.0)}
    h = history()
    h = h[h.id.isin(matches)]
    if len(h):
        sat = h[(h.hour.dt.dayofweek == 5) & h.hour.dt.hour.between(8, 21)]
        busiest = sat.groupby("hour").vph.sum().idxmax()
        obs = sat[sat.hour == busiest][["id", "vph"]]
        out["before"] = calibration(obs, full["baseline"], matches, f"Sat {busiest:%d %b %Y %H:00} (Olgas closed) vs model 'before'")
    lg = hourly(logged(ids=list(matches)))
    if len(lg):
        wk = lg[(lg.hour.dt.dayofweek < 5) & (lg.hour.dt.hour.isin([7, 8, 9, 17, 18, 19])) & (lg.n >= 10)]
        if len(wk):
            obs = wk.groupby("id").vph.mean().reset_index()
            out["after"] = calibration(obs, full["as_built"], matches,
                                       f"weekday peaks {wk.hour.min():%d %b}–{wk.hour.max():%d %b %Y} (Olgas open) vs model 'as built'")
        out["logged"] = {"rows": int(lg.n.sum()), "from": str(lg.hour.min()), "to": str(lg.hour.max())}
    out["stations"] = json.loads(st.assign(matched=st.id.isin(matches), last_time=st.last_time.astype(str)).to_json(drop_id=True))
    (mobility.CASE_DIR / "calibration.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    for k in ("before", "after"):
        if k in out:
            log(f"  {out[k]['label']}: {out[k]['n_detectors']} detectors, observed/model median {out[k]['scale_median']} "
                f"(IQR {out[k]['scale_iqr']}), MAPE {out[k]['mape_pct']}%")
    return out


def _seed0_volumes():
    """Full per-edge car volumes for 'baseline' and 'as_built' (seed 0), same agents as the cached case run."""
    from . import mobility
    c = mobility.cfg()
    nodes, e = mobility.network()
    dests = mobility.destinations(nodes, e)
    rng = np.random.default_rng(100)
    trips = mobility.make_car_trips(nodes, e, rng)
    peds = mobility.make_pedestrians(nodes, e, rng)
    out = {}
    for sid in ("baseline", "as_built"):
        S = mobility.simulate(c["scenarios"][sid], trips, peds, nodes, e, {k: v for k, v in list(dests.items())[:1]})
        out[sid] = S["car_vol"]
    return out


def calibration_result():
    from . import mobility
    p = mobility.CASE_DIR / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else None
