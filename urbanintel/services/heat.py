"""Case study B: urban heat proposals simulated with the agent-based model.

Closes the two gaps of the original pipeline:
  1. Surface temperature now enters route choice: edge heat exposure = 0.5 * (1 - shade) + 0.5 * normalised LST (abm.heat_exposure),
     cost = length * (1 + beta * exposure).
  2. A proposal changes BOTH shade (abm.apply_trees) and surface temperature (tree coefficient of the land-cover response model,
     lulc_model_coefficients.csv), so the land-cover what-if and the ABM act on the same street.
Agents are residents of the top-priority neighbourhoods (pipeline.abm.make_agents); a share are 65+ (slower, same threshold), so the
dose per trip is longer for them. Proposals: street trees sited by abm.select_tree_edges on the streets agents actually use, and
cooling refuges sited by abm.select_refuges. Results are paired by seed and cached in data/outputs/cases/heat/."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import yaml

from pipeline import abm, destinations, landuse, network
from .. import layers as LY

CFG_PATH = LY.ROOT / "urbanintel" / "config" / "cases.yaml"
CASE_DIR = LY.OUT / "cases" / "heat"
EXPOSURE_HIGH = 0.6


def cfg():
    return yaml.safe_load(open(CFG_PATH))["heat"]


def scope_ids(c=None, districts=()):
    """Top-N heat-priority neighbourhoods, city-wide or within the selected municipal districts."""
    c = c or cfg()
    if c.get("neighbourhoods") and not districts:
        return list(c["neighbourhoods"])
    t = pd.read_csv(LY.OUT / "neighbourhood_area_types.csv")
    t = t[t.eligible == True]   # noqa: E712
    if districts:
        t = t[t.district_id.astype(int).isin([int(d) for d in districts])]
    return t.nlargest(c["top_n"], "priority_score").neighbourhood_id.tolist()


def tree_coef():
    c = cfg()
    if c.get("tree_lst_coef_c") is not None:
        return float(c["tree_lst_coef_c"])
    try:
        k = pd.read_csv(LY.OUT / "lulc_model_coefficients.csv")
        k = k[(k.term == "tree") & (k.scope == "municipality")]
        return float(k.estimate.mean())
    except Exception:
        return -3.0


@lru_cache(maxsize=8)
def context(scope: tuple):
    """Street graph around the scope (+ cutoff buffer), edge shade at the configured hour, LST per edge, homes, parks, refuge pool."""
    c = cfg()
    nb = gpd.read_file(LY.ROOT / "data/raw/gis/athens_neighbourhoods.geojson").to_crs(2100)
    nb["neighbourhood_id"] = "athens:nb:" + nb["id"].astype(int).astype(str).str.zfill(3)
    sc = nb[nb.neighbourhood_id.isin(scope)]
    area = sc.geometry.make_valid().union_all().buffer(c["cutoff_m"] + 200)
    streets = gpd.read_file(LY.ROOT / "data/raw/gis/Streets_Center.geojson").to_crs(2100)
    streets = streets[streets.intersects(area)]
    G, _ = network.build_graph(streets, ["raceway", "construction", "proposed", "busway", "elevator"])
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()     # node ids keep indexing the full G.graph["xy"]
    xy = G.graph["xy"]
    # shade by endpoint coordinates (edge_shade.csv was computed on the full-city graph with the same node rounding)
    es = pd.read_csv(LY.OUT / "edge_shade.csv")
    col = f"shade_{int(c['hour']):02d}"
    shade_by_xy = {}
    for x0, y0, x1, y1, s in zip(es.x0.round(1), es.y0.round(1), es.x1.round(1), es.y1.round(1), es[col]):
        k = tuple(sorted([(x0, y0), (x1, y1)]))
        shade_by_xy[k] = s
    keys = [abm.ekey(u, v) for u, v in G.edges()]
    shade = {}
    for u, v in keys:
        k = tuple(sorted([tuple(np.round(xy[u], 1)), tuple(np.round(xy[v], 1))]))
        shade[(u, v)] = float(shade_by_xy.get(k, 0.0))
    # LST at edge midpoints
    mids = [G[u][v]["geom"].interpolate(0.5, normalized=True) for u, v in keys]
    with rasterio.open(LY.OUT / "lst_composite.tif") as src:
        pts = gpd.GeoSeries(mids, crs=2100).to_crs(src.crs)
        lst = np.array([v[0] for v in src.sample(list(zip(pts.x, pts.y)))], float)
    lst[~np.isfinite(lst)] = np.nanmedian(lst)
    jn = gpd.sjoin(gpd.GeoDataFrame({"i": range(len(keys))}, geometry=mids, crs=2100), sc[["neighbourhood_id", "geometry"]],
                   how="left", predicate="within")
    jn = jn[~jn.index.duplicated()]
    jh = gpd.sjoin_nearest(gpd.GeoDataFrame({"i": range(len(keys))}, geometry=mids, crs=2100), streets[["highway", "geometry"]],
                           how="left", max_distance=3)
    jh = jh[~jh.index.duplicated()]
    meta = pd.DataFrame({"length": [G[u][v]["length"] for u, v in keys], "shade": [shade[k] for k in keys], "lst": lst,
                         "nbid": jn.neighbourhood_id.to_numpy(), "highway": jh.highway.fillna("unknown").to_numpy()},
                        index=pd.MultiIndex.from_tuples(keys))
    meta["in_scope"] = meta.nbid.isin(scope)
    # homes
    bp = pd.read_csv(LY.OUT / "building_population.csv")
    bp = bp[bp.neighbourhood_id.isin(scope)]
    node, snap_m, ok = network.snap(G, bp[["x", "y"]].to_numpy(), 150)
    bp = bp.assign(node=node, snap_m=snap_m)[ok]
    # parks and refuge pool
    rules = yaml.safe_load(open(LY.ROOT / "landuse_classes.yaml"))["rules"]
    lu = gpd.read_file(LY.ROOT / "data/raw/gis/landuse.geojson").to_crs(2100)
    lu = lu[lu.intersects(area)]
    lu["geometry"] = lu.geometry.make_valid()
    lu["landuse_class"] = landuse.classify_landuse(lu, rules)
    nodes = destinations.nodes_gdf(G)
    green = [n for n in destinations.green_entry_nodes(lu, nodes, 5000, 25) if n in G]
    pool = [n for n in destinations.refuge_entry_nodes(lu, nodes, entry_m=25) if n in G]
    from . import overview
    stops = overview.vector("bus_stops").to_crs(2100)
    stops = stops[stops.intersects(area)]
    sn, _, sok = network.snap(G, np.c_[stops.geometry.x, stops.geometry.y], 150)
    bus = sorted({int(n) for n in sn[sok] if n in G})
    return {"G": G, "keys": keys, "meta": meta, "bp": bp, "green": green, "pool": pool, "bus": bus,
            "names": sc.set_index("neighbourhood_id")["name"].to_dict()}


def _arrays(ctx, shade, lst, beta):
    keys = ctx["keys"]
    G = ctx["G"]
    u = np.array([k[0] for k in keys]); v = np.array([k[1] for k in keys])
    L = ctx["meta"]["length"].to_numpy()
    s = np.array([shade[k] for k in keys])
    ex = abm.heat_exposure(s, lst)
    eu, ev = np.r_[u, v], np.r_[v, u]
    LL, EX, SUN = np.r_[L, L], np.r_[ex, ex], np.r_[1 - s, 1 - s]
    return eu, ev, LL, EX, SUN, max(G.nodes) + 1


def _route(ctx, agents, shade, lst, beta, dests, speed):
    eu, ev, L, EX, SUN, n = _arrays(ctx, shade, lst, beta)
    cost = L * (1 + beta * EX)
    t = L / speed                        # per-agent speed applied below (time scales with 1/speed)
    r, paths = abm.walk_to_nearest(n, eu, ev, cost, L, {"dose_m": L * EX, "sun_m": L * SUN, "high_m": np.where(EX >= EXPOSURE_HIGH, L, 0.0)},
                                   agents.node.to_numpy(), dests)
    half = len(eu) // 2
    r["edges"] = [None if p is None else np.where(p >= half, p - half, p) for p in paths]
    return r


def _metrics(r, agents, c):
    sp = np.where(agents.vulnerable, c["vulnerable_speed_m_s"], c["walk_speed_m_s"])
    w = agents.weight.to_numpy()
    snap = agents.snap_m.to_numpy()
    ok = r.reached.to_numpy(bool) & (r.length_m.to_numpy() + snap <= c["cutoff_m"])
    time_min = (r.length_m.to_numpy() + snap) / sp / 60
    dose_min = (r.dose_m.to_numpy() + 0.75 * snap) / sp / 60       # home-to-street counted at high exposure (conservative)
    over = ok & (dose_min >= c["heat_threshold_min"])
    sun_min = (r.sun_m.to_numpy() + snap) / sp / 60                  # home-to-street counted as sunny (conservative)

    def grp(mask):
        m = mask & ok
        W, Wr = w[mask].sum(), w[m].sum()
        if Wr == 0:
            return {"heat_min": np.nan, "time_min": np.nan, "sun_min": np.nan, "shade_pct": np.nan, "over_threshold_pct": np.nan,
                    "within5_pct": np.nan, "reached_pct": 0.0}
        tm, sm = float((w[m] * time_min[m]).sum() / Wr), float((w[m] * sun_min[m]).sum() / Wr)
        return {"heat_min": float((w[m] * dose_min[m]).sum() / Wr), "time_min": tm, "sun_min": sm,
                "shade_pct": float(100 * (1 - sm / tm)) if tm else np.nan,
                "over_threshold_pct": float(100 * w[mask & over].sum() / W), "within5_pct": float(100 * w[m & (time_min <= 5)].sum() / W),
                "reached_pct": float(100 * Wr / W)}
    vul = agents.vulnerable.to_numpy(bool)
    out = {f"all_{k}": v for k, v in grp(np.ones(len(w), bool)).items()}
    out.update({f"vuln_{k}": v for k, v in grp(vul).items()})
    return out, ok, dose_min


def simulate(tree_budget_m=1000, refuges=0, scope=None, seeds=None, beta=None, keep_layers=False, dest="cool"):
    """One proposal (street trees + cooling refuges) vs baseline, over seeds. Returns summary dict (+ layers for seed 0).
    dest: 'cool' = walk to the nearest park or cooling refuge; 'bus_stop' = walk to the nearest bus stop (refuges do not apply)."""
    c = cfg()
    scope = tuple(scope or scope_ids(c))
    ctx = context(scope)
    meta, G = ctx["meta"], ctx["G"]
    shade0 = meta["shade"].to_dict()
    lst0 = meta["lst"].to_numpy()
    coef = tree_coef()
    rows, nb_rows, layers = [], [], {}
    for seed in range(seeds or c["seeds"]):
        rng = np.random.default_rng(500 + seed)
        b = beta if beta is not None else rng.uniform(*c["beta_range"])
        ag = abm.make_agents(ctx["bp"], list(scope), c["n_agents_per_neighbourhood"], rng, "pop_cal")
        ag["vulnerable"] = rng.random(len(ag)) < c["vulnerable_share"]
        targets = ctx["bus"] if dest == "bus_stop" else ctx["green"]
        r0 = _route(ctx, ag, shade0, lst0, b, targets, c["walk_speed_m_s"])
        m0, ok0, d0 = _metrics(r0, ag, c)
        usage = {}
        for p, wt in zip(r0.edges, ag.weight):
            if p is not None:
                for i in p:
                    k = ctx["keys"][i]; usage[k] = usage.get(k, 0.0) + wt
        chosen, used = abm.select_tree_edges(meta, tree_budget_m, usage, c["allowed_highway"], 25, 0.6) if tree_budget_m else ([], 0.0)
        shade1, trees, kerb = abm.apply_trees(shade0, chosen, meta["length"].to_dict(), c["canopy_gain"], 10, 6)
        lst1 = lst0.copy()
        if chosen:
            pos = meta.index.get_indexer(pd.MultiIndex.from_tuples(chosen))
            lst1[pos] += coef * c["canopy_gain"] * c["street_width_share"]
        ref = abm.select_refuges(G, ag, np.where(ok0, r0.length_m.to_numpy(), np.nan), ctx["pool"], int(refuges), c["cutoff_m"]) \
            if refuges and dest != "bus_stop" else []
        r1 = _route(ctx, ag, shade1, lst1, b, list(targets) + list(ref), c["walk_speed_m_s"])
        m1, ok1, d1 = _metrics(r1, ag, c)
        rows.append({"seed": seed, "beta": b, "trees": trees, "street_m": used, "kerb_m": kerb, "refuges": len(ref),
                     **{f"{k}_before": v for k, v in m0.items()}, **{f"{k}_after": v for k, v in m1.items()}})
        for nid, ix in ag.groupby("nbid").indices.items():
            a = ag.iloc[ix].reset_index(drop=True)
            mb, _, _ = _metrics(r0.iloc[ix].reset_index(drop=True), a, c)
            ma, _, _ = _metrics(r1.iloc[ix].reset_index(drop=True), a, c)
            nb_rows.append({"seed": seed, "neighbourhood_id": nid, "name": ctx["names"].get(nid, nid), "people": float(a.weight.sum()),
                            **{f"{k}_before": v for k, v in mb.items()}, **{f"{k}_after": v for k, v in ma.items()}})
        if seed == 0 and keep_layers:
            layers = _layers(ctx, chosen, ref, shade0, shade1, lst0, lst1, r0, r1)
    runs = pd.DataFrame(rows)
    mets = [k[:-7] for k in runs.columns if k.endswith("_before")]
    summ = {}
    for k in mets:
        d = runs[f"{k}_after"] - runs[f"{k}_before"]
        summ[k] = {"before": _f(runs[f"{k}_before"].mean()), "after": _f(runs[f"{k}_after"].mean()), "delta": _f(d.mean()),
                   "p05": _f(d.quantile(.05)), "p95": _f(d.quantile(.95))}
    nbd = pd.DataFrame(nb_rows).drop(columns="seed").groupby(["neighbourhood_id", "name"]).mean().reset_index()
    for k in mets:
        nbd[f"{k}_delta"] = nbd[f"{k}_after"] - nbd[f"{k}_before"]
    return {"params": {"tree_budget_m": tree_budget_m, "refuges": refuges, "scope": list(scope)},
            "proposal": {"trees": float(runs.trees.mean()), "street_m": float(runs.street_m.mean()), "kerb_m": float(runs.kerb_m.mean()),
                         "refuges": float(runs.refuges.mean()), "lst_change_on_treated_c": round(coef * c["canopy_gain"] * c["street_width_share"], 3)},
            "summary": summ, "by_neighbourhood": json.loads(nbd.round(3).to_json(orient="records", force_ascii=False)), "layers": layers}


def _layers(ctx, chosen, ref, shade0, shade1, lst0, lst1, r0, r1):
    keys, G, meta = ctx["keys"], ctx["G"], ctx["meta"]
    ex0 = abm.heat_exposure(np.array([shade0[k] for k in keys]), lst0)
    ex1 = abm.heat_exposure(np.array([shade1[k] for k in keys]), lst1)
    use0 = np.zeros(len(keys)); use1 = np.zeros(len(keys))
    for p in r0.edges:
        if p is not None: use0[p] += 1
    for p in r1.edges:
        if p is not None: use1[p] += 1
    sel = meta.in_scope.to_numpy() | (use0 > 0) | (use1 > 0)
    chosen_set = set(chosen)
    g = gpd.GeoDataFrame({"exposure_before": ex0.round(3), "exposure_after": ex1.round(3), "lst_before": lst0.round(2),
                          "lst_after": lst1.round(2), "walkers_before": use0, "walkers_after": use1,
                          "treated": [k in chosen_set for k in keys], "nbid": meta.nbid.to_numpy()},
                         geometry=[G[u][v]["geom"] for u, v in keys], crs=2100)[sel].to_crs(4326)
    xy = G.graph["xy"]
    rf = gpd.GeoDataFrame({"kind": ["refuge"] * len(ref)}, geometry=gpd.points_from_xy([xy[n][0] for n in ref], [xy[n][1] for n in ref]),
                          crs=2100).to_crs(4326)
    return {"edges": json.loads(g.to_json(drop_id=True)), "refuges": json.loads(rf.to_json(drop_id=True))}


def _f(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def _key(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True, default=str).encode()).hexdigest()[:10]


def cached(tree_budget_m=1000, refuges=0, scope=None, force=False, dest="cool"):
    """Simulate once per parameter set; results (with map layers) are cached as JSON."""
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    scope = list(scope or scope_ids())
    if dest == "bus_stop":
        refuges = 0
    extra = {"d": dest} if dest != "cool" else {}
    p = CASE_DIR / f"run_{_key(t=tree_budget_m, r=refuges, s=scope, c=cfg(), v=2, **extra)}.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    res = simulate(tree_budget_m, refuges, scope, keep_layers=True, dest=dest)
    res["params"]["dest"] = dest
    p.write_text(json.dumps(res, ensure_ascii=False))
    return res


def run_case(force=False, log=print):
    """Sweep of the configured budgets (trees alone, refuges alone, combined) for the default scope."""
    c = cfg()
    out = CASE_DIR / "sweep.json"
    if out.exists() and not force:
        return json.loads(out.read_text())
    combos = [(b, 0) for b in c["tree_budgets_m"]] + [(0, k) for k in c["refuge_budgets"] if k] + \
             [(max(c["tree_budgets_m"]), max(c["refuge_budgets"]))]
    rows = []
    for t, k in combos:
        r = cached(t, k, force=force)
        rows.append({"tree_budget_m": t, "refuges": k, **r["proposal"],
                     **{f"{m}_{s}": v[s] for m, v in r["summary"].items() for s in ("before", "after", "delta", "p05", "p95")}})
        log(f"  trees {t} m, refuges {k}: heat dose {r['summary']['all_heat_min']['delta']:+.3f} min/trip, "
            f"65+ over threshold {r['summary']['vuln_over_threshold_pct']['delta']:+.2f} pp")
    names = context(tuple(scope_ids(c)))["names"]
    res = {"case": "heat", "title": c["title"], "scope": scope_ids(c), "scope_names": names, "sweep": rows,
           "assumptions": {k: c[k] for k in ("hour", "vulnerable_share", "vulnerable_speed_m_s", "heat_threshold_min", "canopy_gain",
                                             "street_width_share", "beta_range", "cutoff_m")} | {"tree_lst_coef_c": tree_coef()}}
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res
