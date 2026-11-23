"""Case study A: reopening Vas. Olgas to private cars at Kallimarmaro, with and without street trees.

Agent-based, two agent types on the same OSM network (osmnx, cached):
  cars        -> time-stepped incremental assignment with BPR congestion (pipeline.abm.assign_cars)
  pedestrians -> residents (from the dasymetric building population) walking to parks, bus stops, metro and the stadium; route cost
                 includes sun exposure and the crossing wait created by the car agents' flows (pipeline.abm.crossing_wait)
Scenarios differ only in network supply (Olgas open or not) and shade (trees or not); within a seed every scenario uses the same
agents, so differences are paired. Results are cached in data/outputs/cases/kallimarmaro/."""
from __future__ import annotations

import json
from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from pipeline import abm
from .. import layers as LY

CFG_PATH = LY.ROOT / "urbanintel" / "config" / "cases.yaml"
CASE_DIR = LY.OUT / "cases" / "kallimarmaro"
OSM_DIR = LY.ROOT / "data" / "cache" / "osm"
TO2100 = Transformer.from_crs(4326, 2100, always_xy=True)

DRIVE_HW = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link", "tertiary",
            "tertiary_link", "residential", "unclassified", "living_street", "service"}
CAP_PER_LANE = {"trunk": 1000, "primary": 900, "secondary": 800, "tertiary": 700, "residential": 500, "unclassified": 500,
                "living_street": 300, "service": 300}
SPEED_KMH = {"trunk": 50, "primary": 45, "secondary": 40, "tertiary": 30, "residential": 25, "unclassified": 25,
             "living_street": 15, "service": 15}
DEFAULT_LANES = {"trunk": 3, "primary": 2, "secondary": 2}
NO_WALK = {"motorway", "motorway_link"}
BUS_ONLY_VPH = 40          # buses/taxis on other bus-only streets (e.g. the Amalias bus lanes)
AMALIAS = "Αμαλίας"


def cfg():
    return yaml.safe_load(open(CFG_PATH))["kallimarmaro"]


def _first(x):
    if isinstance(x, list):
        return x[0] if x else None
    if isinstance(x, str) and x.startswith("["):
        try:
            return json.loads(x.replace("'", '"'))[0]
        except Exception:
            return x
    return x


def _num(x, default):
    x = _first(x)
    try:
        f = float(str(x).split()[0])
        return f if np.isfinite(f) else default
    except Exception:
        return default


# ------------------------------------------------------------------------------------------------------------ network
@lru_cache(maxsize=1)
def network():
    """Edge and node tables for the case-study area. Nodes are re-indexed 0..n-1; coordinates in EPSG:2100."""
    import osmnx as ox
    c = cfg()
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    gml = OSM_DIR / "kallimarmaro_all.graphml"
    ox.settings.cache_folder = str(OSM_DIR)
    if not gml.exists():
        ox.settings.useful_tags_way = list(dict.fromkeys(ox.settings.useful_tags_way + ["motor_vehicle", "access", "bus", "psv", "lanes",
                                                                                         "maxspeed", "sidewalk"]))
        G = ox.graph_from_point(tuple(c["centre_latlon"]), dist=c["radius_m"], network_type="all", simplify=True)
        ox.save_graphml(G, gml)
    G = ox.load_graphml(gml)
    nodes, edges = ox.graph_to_gdfs(G)
    nodes = nodes.reset_index()
    nid = {o: i for i, o in enumerate(nodes.osmid)}
    x, y = TO2100.transform(nodes.x.to_numpy(), nodes.y.to_numpy())
    nodes["X"], nodes["Y"] = x, y
    e = edges.reset_index()
    e["iu"], e["iv"] = e.u.map(nid), e.v.map(nid)
    e = e[e.iu != e.iv].reset_index(drop=True)
    for col in ("highway", "name", "motor_vehicle", "access", "bus", "lanes", "maxspeed", "oneway"):
        if col not in e:
            e[col] = None
    e["hw"] = e.highway.map(_first).astype(str)
    e["name_s"] = e.name.map(lambda v: " / ".join(v) if isinstance(v, list) else ("" if v is None or v != v else str(v)))
    mv, acc = e.motor_vehicle.map(_first).astype(str), e.access.map(_first).astype(str)
    e["drivable"] = e.hw.isin(DRIVE_HW) & ~mv.isin(["no", "private"]) & ~acc.isin(["no", "private"])
    e["bus_only"] = ~e.drivable & e.bus.map(_first).astype(str).isin(["designated", "yes"])
    e["olgas"] = e.name_s.str.contains(c["road_name"], regex=False) & ~e.drivable
    lanes_default = e.hw.map(DEFAULT_LANES).fillna(1)
    two_way = ~e.oneway.astype(str).str.lower().isin(["true", "1", "yes"])
    e["lanes_n"] = [max(1.0, _num(l, d) / (2 if tw and _num(l, 0) >= 2 else 1)) for l, d, tw in zip(e.lanes, lanes_default, two_way)]
    e["cap"] = e.hw.map(CAP_PER_LANE).fillna(500) * e.lanes_n
    e["v_kmh"] = [min(_num(ms, SPEED_KMH.get(h, 25)), SPEED_KMH.get(h, 25) + 10) for ms, h in zip(e.maxspeed, e.hw)]
    e["walkable"] = ~e.hw.isin(NO_WALK)
    e = gpd.GeoDataFrame(e, geometry="geometry", crs=4326)
    e["len_m"] = e.to_crs(2100).length
    # signalised junctions: each municipal traffic-light point marks its ONE nearest road junction (>= 3 drivable neighbours) within
    # 25 m; OSM traffic_signals tags count only on such junctions. Mid-block pedestrian lights are therefore not counted as junctions.
    dr = e[e.drivable]
    nb = pd.concat([pd.DataFrame({"a": dr.iu, "b": dr.iv}), pd.DataFrame({"a": dr.iv, "b": dr.iu})]).drop_duplicates()
    junction = np.zeros(len(nodes), bool)
    deg = nb.groupby("a").size()
    junction[deg[deg >= 3].index.to_numpy()] = True
    jid = np.flatnonzero(junction)
    tl = gpd.read_file(LY.ROOT / LY.BY_ID["traffic_lights"].path).to_crs(2100)
    d, j = cKDTree(np.c_[nodes.X.to_numpy()[jid], nodes.Y.to_numpy()[jid]]).query(np.c_[tl.geometry.x, tl.geometry.y])
    sig = np.zeros(len(nodes), bool)
    sig[jid[j[d <= 25]]] = True
    osm_sig = nodes.get("highway", pd.Series(index=nodes.index, dtype=str)).astype(str).to_numpy() == "traffic_signals"
    nodes["signal"] = sig | (osm_sig & junction)
    e["shade0"] = _edge_shade(e)
    e["uturn"] = False
    # ends of Vas. Olgas: nodes where an Olgas segment meets the drivable network (new signals in the as-built project)
    ol = e[e.olgas]
    touch = set(e.iu[e.drivable]) | set(e.iv[e.drivable])
    nodes["olgas_end"] = nodes.index.isin([n for n in set(ol.iu) | set(ol.iv) if n in touch])
    e = _add_uturns(nodes, e, cfg())
    return nodes, e


def _add_uturns(nodes, e, c):
    """Restored U-turn on Amalias at Olgas: one connector each way between the two carriageways, nearest node pair within search_m.
    The connectors are only usable in scenarios with works=true."""
    u = c.get("amalias_uturn")
    if not u:
        return e
    x, y = TO2100.transform(u["point_latlon"][1], u["point_latlon"][0])
    X, Y = nodes.X.to_numpy(), nodes.Y.to_numpy()
    am = e[e.drivable & e.name_s.str.contains(AMALIAS, regex=False)]
    dy = Y[am.iv.to_numpy()] - Y[am.iu.to_numpy()]
    north, south = am[dy > 0], am[dy < 0]

    def near(ids):
        ids = np.unique(np.asarray(ids))
        return ids[np.hypot(X[ids] - x, Y[ids] - y) <= u["search_m"]]
    rows = []
    for arrive, leave, lab in ((near(south.iv), near(north.iu), "U-turn S→N (Amalias)"), (near(north.iv), near(south.iu), "U-turn N→S (Amalias)")):
        if not len(arrive) or not len(leave):
            continue
        d = np.hypot(X[arrive][:, None] - X[leave][None, :], Y[arrive][:, None] - Y[leave][None, :])
        i, j = np.unravel_index(np.argmin(d), d.shape)
        a, b = int(arrive[i]), int(leave[j])
        if a == b:
            continue
        geom = LineString([(nodes.x[a], nodes.y[a]), (nodes.x[b], nodes.y[b])])
        rows.append({"iu": a, "iv": b, "hw": "uturn", "name_s": lab, "drivable": False, "olgas": False, "bus_only": False, "walkable": False,
                     "lanes_n": 1.0, "cap": 600.0, "v_kmh": 10.0, "len_m": float(d[i, j]) + 10.0, "shade0": 0.0, "uturn": True, "geometry": geom})
    if rows:
        e = gpd.GeoDataFrame(pd.concat([e, gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)], ignore_index=True), geometry="geometry", crs=4326)
    return e


def _edge_shade(e):
    """Shade share per edge at 14:00: canopy (Copernicus tree cover density along the edge) combined with building shade
    (pipeline edge_shade.csv, nearest street-centre edge within 12 m)."""
    g = e.to_crs(3035)
    pts, owner = [], []
    for i, ln in enumerate(g.geometry):
        n = max(2, int(ln.length // 10))
        for f in np.linspace(0, 1, n):
            p = ln.interpolate(f, normalized=True)
            pts.append((p.x, p.y)); owner.append(i)
    with rasterio.open(LY.ROOT / LY.BY_ID["tree_density"].path) as src:
        vals = np.array([v[0] for v in src.sample(pts)], float)
        vals[vals == src.nodata] = 0
    tree = pd.Series(vals / 100).groupby(np.array(owner)).mean().reindex(range(len(e))).fillna(0).to_numpy()
    bld = np.zeros(len(e))
    try:
        es = pd.read_csv(LY.OUT / "edge_shade.csv", usecols=["x0", "y0", "x1", "y1", "shade_14"])
        tree_xy = np.c_[(es.x0 + es.x1) / 2, (es.y0 + es.y1) / 2]
        mid = e.to_crs(2100).geometry.interpolate(0.5, normalized=True)
        d, j = cKDTree(tree_xy).query(np.c_[mid.x, mid.y])
        bld = np.where(d <= 12, es.shade_14.to_numpy()[j], 0.0)
    except Exception:
        pass
    return 1 - (1 - np.clip(tree, 0, 1)) * (1 - np.clip(bld, 0, 1))


# ------------------------------------------------------------------------------------------------------------ agents
def _snap(nodes_xy, pts):
    d, i = cKDTree(nodes_xy).query(pts)
    return i, d


def _points_to_nodes(nodes, mask, latlon):
    idx = np.flatnonzero(mask)
    x, y = TO2100.transform([p[1] for p in latlon], [p[0] for p in latlon])
    j, _ = _snap(np.c_[nodes.X.to_numpy()[idx], nodes.Y.to_numpy()[idx]], np.c_[x, y])
    return idx[j]


def _near_nodes(nodes, mask, latlon, radius):
    idx = np.flatnonzero(mask)
    x, y = TO2100.transform([p[1] for p in latlon], [p[0] for p in latlon])
    tree = cKDTree(np.c_[nodes.X.to_numpy()[idx], nodes.Y.to_numpy()[idx]])
    hit = sorted({k for lst in tree.query_ball_point(np.c_[x, y], radius) for k in lst})
    return idx[hit] if hit else _points_to_nodes(nodes, mask, latlon)


@lru_cache(maxsize=1)
def residents():
    """Buildings with modelled residents inside the pedestrian radius (EPSG:2100)."""
    c = cfg()
    bp = pd.read_csv(LY.OUT / "building_population.csv", usecols=["x", "y", "neighbourhood_id", "pop_cal"])
    cx, cy = TO2100.transform(c["centre_latlon"][1], c["centre_latlon"][0])
    r = np.hypot(bp.x - cx, bp.y - cy)
    return bp[(r <= c["radius_m"] - 200) & (bp.pop_cal > 0)].assign(r=r).reset_index(drop=True)


def destinations(nodes, e):
    c = cfg()
    walk_nodes = np.zeros(len(nodes), bool)
    walk_nodes[np.r_[e.iu[e.walkable], e.iv[e.walkable]]] = True
    xy = np.c_[nodes.X, nodes.Y]
    out = {}
    parks = gpd.read_file(LY.ROOT / LY.BY_ID["parks"].path).to_crs(2100)
    parks = parks[parks.area >= 5000]
    near = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2100)[walk_nodes],
                     gpd.GeoDataFrame(geometry=parks.buffer(25), crs=2100), predicate="within")
    out["park"] = np.unique(near.index.to_numpy())
    ng = parks[parks["name"].astype(str).isin(c["national_garden_names"])]
    gd = gpd.read_file(LY.ROOT / LY.BY_ID["gardens"].path).to_crs(2100)
    ng = pd.concat([ng[["geometry"]], gd[gd["name"].astype(str).isin(c["national_garden_names"])][["geometry"]]])
    if len(ng):
        near = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2100)[walk_nodes],
                         gpd.GeoDataFrame(geometry=ng.buffer(25), crs=2100), predicate="within")
        out["national_garden"] = np.unique(near.index.to_numpy())
    stops = gpd.read_file(LY.ROOT / LY.BY_ID["bus_stops"].path).to_crs(2100)
    cx, cy = TO2100.transform(c["centre_latlon"][1], c["centre_latlon"][0])
    stops = stops[np.hypot(stops.geometry.x - cx, stops.geometry.y - cy) <= c["radius_m"]]
    out["bus_stop"] = np.unique(_points_to_nodes(nodes, walk_nodes, [(p.y, p.x) for p in stops.to_crs(4326).geometry]))
    for k, pts in c["destinations"].items():
        out[k] = np.unique(_points_to_nodes(nodes, walk_nodes, pts))
    return out


def make_car_trips(nodes, e, rng):
    c, cc = cfg(), cfg()["cars"]
    car = e[e.drivable | e.olgas]
    has_out = np.zeros(len(nodes), bool); has_out[e.iu[e.drivable]] = True
    has_in = np.zeros(len(nodes), bool); has_in[e.iv[e.drivable]] = True
    cx, cy = TO2100.transform(c["centre_latlon"][1], c["centre_latlon"][0])
    rn = np.hypot(nodes.X - cx, nodes.Y - cy).to_numpy()
    major = car[car.hw.isin(["trunk", "primary", "secondary"])]
    gw_cap = pd.concat([major.groupby("iu").cap.sum(), major.groupby("iv").cap.sum()]).groupby(level=0).sum()
    gw = gw_cap[(rn[gw_cap.index] >= c["radius_m"] - 300)]
    gw_o = gw[has_out[gw.index]]; gw_d = gw[has_in[gw.index]]
    res = residents()
    rnode, _ = _snap(np.c_[nodes.X, nodes.Y][has_out & has_in], res[["x", "y"]].to_numpy())
    rnode = np.flatnonzero(has_out & has_in)[rnode]
    att = _points_to_nodes(nodes, has_out & has_in, c["attractors"])
    n = cc["n_agents"]
    n_cor = int(n * cc.get("corridor_share", 0))
    co, cd = _near_nodes(nodes, has_out & has_in, c["corridor"]["origins"], c["corridor"]["radius_m"]), \
        _near_nodes(nodes, has_out & has_in, c["corridor"]["destinations"], c["corridor"]["radius_m"])
    n -= n_cor
    n_th = int(n * cc["through_share"])
    o = rng.choice(gw_o.index.to_numpy(), n_th, p=(gw_o / gw_o.sum()).to_numpy())
    d = rng.choice(gw_d.index.to_numpy(), n_th, p=(gw_d / gw_d.sum()).to_numpy())
    far = np.hypot(nodes.X.to_numpy()[o] - nodes.X.to_numpy()[d], nodes.Y.to_numpy()[o] - nodes.Y.to_numpy()[d]) >= 1500
    o, d = o[far], d[far]
    n_loc = n - len(o)
    hp = res.pop_cal.to_numpy() / res.pop_cal.sum()
    homes = rnode[rng.choice(len(res), n_loc, p=hp)]
    others = np.r_[att, gw_d.index.to_numpy()]
    away = rng.choice(others, n_loc)
    outbound = rng.random(n_loc) < 0.5
    o2, d2 = np.where(outbound, homes, away), np.where(outbound, away, homes)
    o3, d3 = rng.choice(co, n_cor), rng.choice(cd, n_cor)
    back = rng.random(n_cor) < cc.get("corridor_reverse_share", 0.0)    # the same movement in the other direction
    o3, d3 = np.where(back, d3, o3), np.where(back, o3, d3)
    trips = pd.DataFrame({"o": np.r_[o, o2, o3], "d": np.r_[d, d2, d3],
                          "kind": ["through"] * len(o) + ["local"] * n_loc + ["corridor"] * n_cor})
    trips = trips[trips.o != trips.d].reset_index(drop=True)
    trips["weight"] = cc["peak_trips_vph"] * cc.get("demand_scale", 1.0) / len(trips)
    trips["step"] = rng.integers(0, cc["steps"], len(trips))
    return trips


def make_pedestrians(nodes, e, rng):
    c = cfg()
    res = residents()
    res = res[res.r <= c["ped_radius_m"]]
    walk_nodes = np.zeros(len(nodes), bool)
    walk_nodes[np.r_[e.iu[e.walkable], e.iv[e.walkable]]] = True
    idx = np.flatnonzero(walk_nodes)
    pick = rng.choice(len(res), c["pedestrians"]["n_agents"], p=(res.pop_cal / res.pop_cal.sum()).to_numpy())
    s = res.iloc[pick]
    j, d = _snap(np.c_[nodes.X.to_numpy()[idx], nodes.Y.to_numpy()[idx]], s[["x", "y"]].to_numpy())
    ol = e[e.olgas].to_crs(2100).geometry.union_all()
    dist_ol = gpd.GeoSeries(gpd.points_from_xy(s.x, s.y), crs=2100).distance(ol).to_numpy() if not ol.is_empty else np.full(len(s), np.inf)
    return pd.DataFrame({"node": idx[j], "snap_m": d, "nbid": s.neighbourhood_id.to_numpy(),
                         "corridor": dist_ol <= c["corridor_residents_m"], "weight": res.pop_cal.sum() / len(s)})


# ------------------------------------------------------------------------------------------------------------ one scenario
def simulate(scn, trips, peds, nodes, e, dests):
    c = cfg(); cc, pc = c["cars"], c["pedestrians"]
    n = len(nodes)
    # --- supply
    works = bool(scn.get("works", scn["olgas_open"]))
    ol = e.olgas.to_numpy()
    car = (e.drivable | (e.olgas & scn["olgas_open"]) | (e.uturn & works)).to_numpy()
    cap = e.cap.to_numpy().copy(); v = e.v_kmh.to_numpy().copy()
    if scn["olgas_open"]:
        cap[ol] = c["reopened"]["capacity_vph_per_lane"]; v[ol] = c["reopened"]["speed_kmh"]
    sig = nodes.signal.to_numpy().copy()
    if works and c.get("new_signal_ends"):
        sig |= nodes.olgas_end.to_numpy()
    t0 = e.len_m.to_numpy() / (v / 3.6) + np.where(sig[e.iv.to_numpy()], cc["signal_delay_s"], 0.0)
    ci = np.flatnonzero(car)
    A = abm.assign_cars(n, e.iu.to_numpy()[ci], e.iv.to_numpy()[ci], t0[ci], cap[ci], trips, cc["steps"], cc["bpr_alpha"], cc["bpr_power"])
    car_vol = np.zeros(len(e)); car_vol[ci] = A["volume"]
    # buses/tram: other bus-only streets keep their buses; Olgas gets buses + tram back only with the project (none during closure)
    bus = np.where(e.bus_only.to_numpy() & ~ol, BUS_ONLY_VPH, 0.0) + np.where(ol & works, c["reopened"].get("bus_tram_vph", 0), 0.0)
    vol = car_vol + bus
    # --- street flow (both directions of the same street) and crossing wait at each node
    key = pd.Series(list(zip(np.minimum(e.iu, e.iv), np.maximum(e.iu, e.iv))))
    link_flow = pd.Series(vol).groupby(key).transform("sum").to_numpy()
    lanes_total = pd.Series(np.where(vol > 0, e.lanes_n, 0)).groupby(key).transform("sum").to_numpy()
    node_flow = np.zeros(n); node_w = np.zeros(n)
    for col in ("iu", "iv"):
        ix = e[col].to_numpy()
        np.maximum.at(node_flow, ix, link_flow)
        np.maximum.at(node_w, ix, lanes_total * pc["lane_width_m"])
    wait_node = abm.crossing_wait(node_flow, sig, np.maximum(node_w, pc["lane_width_m"]), pc["signal_cycle_s"], pc["ped_green_s"])
    # --- shade
    shade = e.shade0.to_numpy().copy()
    if scn["trees"]:
        tm = e.name_s.apply(lambda s: any(t in s for t in c["tree_streets"])).to_numpy()
        shade[tm] = shade[tm] + c["tree_canopy_gain"] * (1 - shade[tm])
    # --- pedestrian graph: walkable edges in both directions
    w = e[e.walkable]
    wi = w.index.to_numpy()
    eu = np.r_[w.iu, w.iv]; ev = np.r_[w.iv, w.iu]; src = np.r_[wi, wi]
    L = e.len_m.to_numpy()[src]
    t = L / pc["speed_m_s"]
    sun = t * (1 - shade[src])
    wait = pc["crossing_need"] * wait_node[ev]
    traffic_m = np.where(link_flow[src] >= 300, L, 0.0)
    cost = t * (1 + pc["beta_sun"] * (1 - shade[src])) + wait
    ped = {}
    for dk, dn in dests.items():
        r, paths = abm.walk_to_nearest(n, eu, ev, cost, L, {"time_s": t, "sun_s": sun, "wait_s": wait, "traffic_m": traffic_m},
                                       peds.node.to_numpy(), dn)
        snap_t = peds.snap_m.to_numpy() / pc["speed_m_s"]
        r["total_min"] = (r.time_s + r.wait_s + snap_t) / 60
        r["reached"] = r.reached & (r.total_min <= pc["cutoff_min"])
        r["edges"] = [None if p is None else src[p] for p in paths]
        ped[dk] = r
    return {"car": A, "car_mask": car, "vol": vol, "car_vol": car_vol, "link_flow": link_flow, "shade": shade, "wait_node": wait_node,
            "ped": ped, "signal": sig}


def _ped_metrics(r, wgt):
    ok = r.reached.to_numpy(bool)
    W = wgt.sum()
    wr = wgt[ok].sum()
    def m(col):
        return float(np.nansum(wgt[ok] * r[col].to_numpy()[ok]) / wr) if wr else np.nan
    return {"time_min": m("total_min"), "sun_min": m("sun_s") / 60, "wait_min": m("wait_s") / 60, "traffic_m": m("traffic_m"),
            "within5_pct": 100 * wgt[ok & (r.total_min.to_numpy() <= 5)].sum() / W,
            "within10_pct": 100 * wgt[ok & (r.total_min.to_numpy() <= 10)].sum() / W, "reached_pct": 100 * wr / W}


def _car_metrics(A, trips, e, car):
    tt = A["trip_time"]; ok = np.isfinite(tt); w = trips.weight.to_numpy()
    vol = A["volume"]; ci = np.flatnonzero(car)
    L = e.len_m.to_numpy()[ci]
    vkt = float((vol * L).sum() / 1000)
    vht = float((vol * A["time"]).sum() / 3600)
    return {"trip_min": float((w[ok] * tt[ok]).sum() / w[ok].sum() / 60), "veh_km": vkt, "veh_hours": vht,
            "mean_speed_kmh": vkt / vht if vht else np.nan, "unrouted_pct": float(100 * (~ok).mean())}


# ------------------------------------------------------------------------------------------------------------ the case
def run_case(force=False, seeds=None, log=print):
    out = CASE_DIR / "summary.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    c = cfg()
    nodes, e = network()
    dests = destinations(nodes, e)
    log(f"  network: {len(nodes)} nodes, {len(e)} edges, {int(e.drivable.sum())} drivable, {int(e.olgas.sum())} Olgas segments, "
        f"{int(nodes.signal.sum())} signalised nodes; destinations " + ", ".join(f"{k}={len(v)}" for k, v in dests.items()))
    seeds = seeds or c["seeds"]
    street_masks = _street_masks(e, c)
    rows, keep, nb_rows = [], {}, []
    for seed in range(seeds):
        rng = np.random.default_rng(100 + seed)
        trips = make_car_trips(nodes, e, rng)
        peds = make_pedestrians(nodes, e, rng)
        wgt = peds.weight.to_numpy()
        for sid, scn in c["scenarios"].items():
            S = simulate(scn, trips, peds, nodes, e, dests)
            row = {"seed": seed, "scenario": sid, **{f"car_{k}": v for k, v in _car_metrics(S["car"], trips, e, S["car_mask"]).items()},
                   "olgas_vph": float(S["car_vol"][e.olgas.to_numpy()].max()) if e.olgas.any() else 0.0,
                   "uturn_vph": float(S["car_vol"][e.uturn.to_numpy()].sum()) if e.uturn.any() else 0.0}
            for short, mk in street_masks.items():
                row[f"street_{short}_vph"] = float(np.average(S["car_vol"][mk], weights=e.len_m.to_numpy()[mk])) if mk.any() else np.nan
            cor = peds.corridor.to_numpy()
            kinds = trips.kind.to_numpy()
            for kind in ("corridor", "through", "local"):
                km = kinds == kind
                tt = S["car"]["trip_time"][km]
                row[f"car_{kind}_trip_min"] = float(np.nanmean(tt) / 60) if np.isfinite(tt).any() else np.nan
            olg = set(np.flatnonzero(e.olgas.to_numpy()))
            car_on_olgas = row["olgas_vph"]
            walkers = 0.0
            for dk, r in S["ped"].items():
                use = np.array([p is not None and len(olg.intersection(p)) > 0 for p in r.edges])
                walkers += float(wgt[use].sum())
                row.update({f"{dk}_{k}": v for k, v in _ped_metrics(r, wgt).items()})
                if cor.any():
                    row.update({f"corr_{dk}_{k}": v for k, v in _ped_metrics(r[cor].reset_index(drop=True), wgt[cor]).items()})
                for nbid, ix in peds.groupby("nbid").indices.items():
                    m = _ped_metrics(r.iloc[ix].reset_index(drop=True), wgt[ix])
                    nb_rows.append({"seed": seed, "scenario": sid, "dest": dk, "neighbourhood_id": nbid, "people": float(wgt[ix].sum()), **m})
            row["olgas_walk_trips"] = walkers
            row["olgas_walk_trips_with_traffic"] = walkers if car_on_olgas >= 50 else 0.0     # Olgas carries car traffic
            rows.append(row)
            if seed == 0:
                keep[sid] = (S, trips, peds)
        log(f"  seed {seed + 1}/{seeds} done")
    runs = pd.DataFrame(rows)
    runs.to_csv(CASE_DIR / "runs.csv", index=False)
    metrics = [col for col in runs.columns if col not in ("seed", "scenario")]
    base = runs[runs.scenario == "baseline"].set_index("seed")
    summ = {}
    for sid in c["scenarios"]:
        r = runs[runs.scenario == sid].set_index("seed")
        d = r[metrics] - base[metrics]
        summ[sid] = {k: {"mean": _f(r[k].mean()), "delta": _f(d[k].mean()), "p05": _f(d[k].quantile(.05)), "p95": _f(d[k].quantile(.95))}
                     for k in metrics}
    nb = pd.DataFrame(nb_rows).groupby(["scenario", "dest", "neighbourhood_id"]).mean(numeric_only=True).reset_index()
    nb.drop(columns="seed").round(3).to_csv(CASE_DIR / "by_neighbourhood.csv", index=False)
    _write_layers(keep, nodes, e)
    result = {"case": "kallimarmaro", "title": c["title"], "note": c["note"], "seeds": seeds, "sources": c.get("sources", []),
              "forecast_decongestion_pct": c.get("forecast_decongestion_pct"),
              "corridor_streets": {k: int(m.sum()) for k, m in street_masks.items()},
              "scenarios": {k: v["label"] for k, v in c["scenarios"].items()}, "summary": summ,
              "assumptions": {"cars": c["cars"], "pedestrians": c["pedestrians"], "reopened": c["reopened"],
                              "tree_canopy_gain": c["tree_canopy_gain"], "tree_streets": c["tree_streets"], "tree_count": c.get("tree_count"),
                              "new_signal_ends": c.get("new_signal_ends"), "amalias_uturn": bool(c.get("amalias_uturn"))},
              "network": {"nodes": int(len(nodes)), "edges": int(len(e)), "olgas_segments": int(e.olgas.sum()),
                          "signalised_nodes": int(nodes.signal.sum()), "new_signal_nodes": int(nodes.olgas_end.sum()),
                          "uturn_connectors": int(e.uturn.sum()), "destinations": {k: int(len(v)) for k, v in dests.items()}}}
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _street_masks(e, c):
    """Car edges of each watched street (Amalias, Vas. Konstantinou, Arditou) within corridor_streets_radius_m of Vas. Olgas,
    plus 'all' of them together."""
    olg = e[e.olgas].to_crs(2100).geometry.union_all()
    mid = e.to_crs(2100).geometry.interpolate(0.5, normalized=True)
    close = (mid.distance(olg) <= c.get("corridor_streets_radius_m", 700)).to_numpy() & e.drivable.to_numpy()
    out = {}
    for name in c.get("corridor_streets", []):
        short = {"Βασιλίσσης Αμαλίας": "amalias", "Βασιλέως Κωνσταντίνου": "konstantinou", "Αρδηττού": "arditou"}.get(name, name)
        out[short] = close & e.name_s.str.contains(name, regex=False).to_numpy()
    if out:
        out["all"] = np.any(np.vstack(list(out.values())), axis=0)
    return out


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _write_layers(keep, nodes, e):
    """Seed-0 layers for the map: car volume and shade per scenario on every edge, and a sample of agent routes."""
    g = e[["name_s", "hw", "olgas", "uturn", "geometry"]].copy()
    for sid, (S, *_rest) in keep.items():
        g[f"vol_{sid}"] = np.round(S["vol"], 0)
        g[f"shade_{sid}"] = np.round(S["shade"], 3)
    base = g["vol_baseline"]
    for sid in keep:
        g[f"dvol_{sid}"] = g[f"vol_{sid}"] - base
    g = g[(g.filter(like="vol_").abs().sum(axis=1) > 0) | g.olgas | (g.filter(like="shade_").nunique(axis=1) > 1)]
    g.rename(columns={"name_s": "name", "hw": "highway"}).to_file(CASE_DIR / "edges.geojson", driver="GeoJSON")
    feats = []
    geoms = e.geometry.to_numpy()
    for sid in ("baseline", "as_built", "traffic_only"):
        if sid not in keep:
            continue
        S, trips, peds = keep[sid]
        for i, p in enumerate(S["car"]["paths"][:80]):
            if p:
                ci = np.flatnonzero(S["car_mask"])[p]
                feats.append({"scenario": sid, "mode": "car", "agent": i, "geometry": _join(geoms[ci])})
        r = S["ped"]["metro"]
        for i, p in enumerate(r.edges[:80]):
            if p is not None and len(p):
                feats.append({"scenario": sid, "mode": "walk", "agent": i, "geometry": _join(geoms[p], reverse_ok=True)})
    gpd.GeoDataFrame(feats, geometry="geometry", crs=4326).to_file(CASE_DIR / "traces.geojson", driver="GeoJSON")
    olg = e[e.olgas]
    if len(olg):
        olg[["name_s", "geometry"]].to_file(CASE_DIR / "olgas.geojson", driver="GeoJSON")
    ends = nodes[nodes.olgas_end]
    gpd.GeoDataFrame({"kind": "new signal"}, index=ends.index, geometry=gpd.points_from_xy(ends.x, ends.y), crs=4326) \
        .to_file(CASE_DIR / "signals.geojson", driver="GeoJSON")


def _join(lines, reverse_ok=False):
    """Concatenate edge geometries into one ordered line (flipping pieces whose ends do not meet)."""
    pts = []
    for ln in lines:
        c = list(ln.coords)
        if pts:
            if Point(c[-1]).distance(Point(pts[-1])) < Point(c[0]).distance(Point(pts[-1])):
                c = c[::-1]
        pts.extend(c)
    return LineString(pts) if len(pts) > 1 else LineString(pts * 2)


# ------------------------------------------------------------------------------------------------------------ reading
def result():
    return run_case(force=False)


def layer(name):
    """'edges' | 'traces' | 'olgas' | 'signals' as GeoJSON dict."""
    p = CASE_DIR / f"{name}.geojson"
    if not p.exists():
        run_case()
    return json.loads(p.read_text())


def by_neighbourhood():
    p = CASE_DIR / "by_neighbourhood.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()
