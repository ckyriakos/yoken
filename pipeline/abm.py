"""Agent-based micro-simulation of walking trips to cooling destinations, and scenario operators on the street network.
Agents are synthetic residents drawn from modelled building populations. Each walks from home to the destination with the lowest
heat-weighted cost: cost = length * (1 + beta * (1 - shade)). Agents do NOT interact (no crowding, parking search or capacity):
the data for that is not available. Outputs are shade/time-in-sun PROXIES from a building-shadow model, not human thermal exposure."""
from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd


def ekey(u, v):
    return (u, v) if u < v else (v, u)


def set_edge_costs(G, shade, beta):
    """Write 'shade' and heat-weighted 'cost' on every edge. shade: dict ekey -> share in [0, 1]."""
    for u, v, d in G.edges(data=True):
        s = min(1.0, max(0.0, shade.get(ekey(u, v), 0.0)))
        d["shade"] = s
        d["cost"] = d["length"] * (1.0 + beta * (1.0 - s))


def make_agents(bp, nb_ids, n_per_nb, rng, pop_col="pop_cal"):
    """Sample n_per_nb homes per neighbourhood with probability proportional to building population. bp needs columns
    neighbourhood_id, node, snap_m, pop_col. Each agent represents (neighbourhood population / n_per_nb) people."""
    rows = []
    for nid in nb_ids:
        b = bp[(bp.neighbourhood_id == nid) & (bp[pop_col] > 0)]
        if b.empty:
            continue
        p = b[pop_col].to_numpy() / b[pop_col].sum()
        pick = rng.choice(len(b), size=n_per_nb, p=p)
        s = b.iloc[pick]
        rows.append(pd.DataFrame({"nbid": nid, "node": s.node.to_numpy(), "snap_m": s.snap_m.to_numpy(),
                                  "weight": b[pop_col].sum() / n_per_nb}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["nbid", "node", "snap_m", "weight"])


def route_metrics(G, agents, dest, cutoff_m, beta_max=3.0):
    """Heat-weighted shortest path from each agent to its best destination. Returns (per-agent DataFrame, edge usage dict).
    reached means route length + snap distance <= cutoff_m."""
    dest = [d for d in set(dest) if d in G]
    n = len(agents)
    res = pd.DataFrame({"route_m": np.nan, "shaded_m": np.nan, "shortest_m": np.nan, "reached": False}, index=range(n))
    usage = defaultdict(float)
    if not dest or n == 0:
        return res, usage
    _, paths = nx.multi_source_dijkstra(G, dest, weight="cost", cutoff=cutoff_m * (1 + beta_max))
    short = nx.multi_source_dijkstra_path_length(G, dest, weight="length", cutoff=cutoff_m)
    for i, a in enumerate(agents.itertuples()):
        p = paths.get(a.node)
        if p is None:
            continue
        L = S = 0.0
        for u, v in zip(p[:-1], p[1:]):
            d = G[u][v]
            L += d["length"]; S += d["length"] * d["shade"]
        if L + a.snap_m <= cutoff_m:
            res.loc[i, ["route_m", "shaded_m", "shortest_m", "reached"]] = [L, S, short.get(a.node, L), True]
            for u, v in zip(p[:-1], p[1:]):
                usage[ekey(u, v)] += a.weight
    return res, usage


def summarise(res, agents, speed):
    """Population-weighted metrics. Snap distance (home to street) is counted as walking in the sun (conservative)."""
    w, r = agents.weight.to_numpy(), res.reached.to_numpy(bool)
    snap = agents.snap_m.to_numpy()
    route, shaded, short = res.route_m.to_numpy(float), res.shaded_m.to_numpy(float), res.shortest_m.to_numpy(float)
    total_m = np.where(r, route + snap, np.nan)
    sun_m = np.where(r, route - shaded + snap, np.nan)
    W = w.sum()
    wr = w[r].sum()
    return {"reached_pct": 100 * wr / W,
            "within5_pct": 100 * w[r & (total_m <= 5 * 60 * speed)].sum() / W,
            "time_min": float(np.nansum(w * total_m) / wr / speed / 60) if wr else np.nan,
            "sun_min": float(np.nansum(w * sun_m) / wr / speed / 60) if wr else np.nan,
            "shade_share_pct": float(100 * np.nansum(w * shaded) / np.nansum(w * route)) if wr else np.nan,
            "detour_ratio": float(np.nansum(w * route) / np.nansum(w * short)) if wr else np.nan}


def select_tree_edges(meta, budget_m, usage, allowed, min_len, max_shade):
    """Greedy street-tree siting. meta: DataFrame (index = ekey) with length, shade, nbid, highway, in_scope.
    Priority 1: edges used by simulated trips, ranked by usage x length x (1 - shade). Priority 2: remaining eligible edges by length x (1 - shade).
    Whole edges only, until budget_m of street is treated."""
    m = meta[meta.in_scope & meta.highway.isin(allowed) & (meta.length >= min_len) & (meta.shade <= max_shade)].copy()
    m["usage"] = [usage.get(k, 0.0) for k in m.index]
    m["gain"] = m.length * (1 - m.shade)
    m["tier"] = (m.usage <= 0).astype(int)
    m["score"] = np.where(m.usage > 0, m.usage * m.gain, m.gain)
    chosen, used = [], 0.0
    for k, r in m.sort_values(["tier", "score"], ascending=[True, False]).iterrows():
        if used + r.length <= budget_m:
            chosen.append(k); used += r.length
    return chosen, used


def apply_trees(shade, edges, length_of, gain, spacing_m, kerb_m_per_pit):
    """Raise shade on treated edges: s' = s + gain * (1 - s). Returns (new shade dict, trees, kerb metres affected).
    Kerb is an upper bound: it assumes every tree pit replaces kerb-side space of kerb_m_per_pit."""
    new = dict(shade)
    trees = 0
    for k in edges:
        new[k] = new.get(k, 0.0) + gain * (1 - new.get(k, 0.0))
        trees += int(length_of[k] // spacing_m)
    return new, trees, trees * kerb_m_per_pit


def select_refuges(G, agents, base_len, candidates, k, cutoff_m, max_candidates=300):
    """Greedy: each step adds the candidate node that cuts the most population-weighted walking distance (length, capped at cutoff_m).
    base_len: per-agent baseline route length (NaN where unreached)."""
    best = np.where(np.isnan(base_len), cutoff_m, np.minimum(base_len, cutoff_m))
    nodes, w = agents.node.to_numpy(), agents.weight.to_numpy()
    cand = list(candidates)[:max_candidates]
    cache = {c: nx.single_source_dijkstra_path_length(G, c, cutoff=cutoff_m, weight="length") for c in cand if c in G}
    chosen = []
    for _ in range(k):
        bestc, bestg = None, 0.0
        for c, d in cache.items():
            if c in chosen:
                continue
            dc = np.array([d.get(n, cutoff_m) for n in nodes])
            g = float((w * (best - np.minimum(best, dc))).sum())
            if g > bestg:
                bestc, bestg = c, g
        if bestc is None:
            break
        chosen.append(bestc)
        d = cache[bestc]
        best = np.minimum(best, np.array([d.get(n, cutoff_m) for n in nodes]))
    return chosen


# =============================================================================================================================
# Multimodal, time-stepped extension (used by urbanintel case studies).
# Graphs are held as arrays (CSR via scipy) so thousands of agents x time steps x scenarios run in seconds.
#   Car agents: incremental (time-stepped) assignment. The peak hour is split into K steps; in step k the agents departing in that
#   step take the fastest route under the CURRENT congested travel times; times then update with the BPR link-delay function.
#   Congestion and diversion therefore emerge from agents' choices (cars react to each other through link travel time).
#   Pedestrian agents: route on generalised walking cost = time * (1 + beta * (1 - shade)) + crossing wait at the node entered,
#   where the wait depends on the car flow produced by the car agents (signalised: fixed cycle wait; unsignalised: gap acceptance).
#   This is the coupling between the two agent types.
# =============================================================================================================================
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as _sp_dijkstra


def bpr(t0, volume, capacity, alpha=0.15, power=4.0):
    """Bureau of Public Roads link travel time: t = t0 * (1 + alpha * (v / c) ** power)."""
    c = np.maximum(np.asarray(capacity, float), 1e-6)
    return np.asarray(t0, float) * (1.0 + alpha * (np.asarray(volume, float) / c) ** power)


def _csr(n, u, v, w):
    """Directed sparse graph keeping the cheapest of parallel edges. Returns (matrix, dict (u, v) -> edge index)."""
    order = np.lexsort((w, v, u))
    u_, v_, w_ = u[order], v[order], w[order]
    first = np.ones(len(u_), bool)
    first[1:] = (u_[1:] != u_[:-1]) | (v_[1:] != v_[:-1])
    idx = {(a, b): int(order[i]) for i, (a, b) in enumerate(zip(u_, v_)) if first[i]}
    m = csr_matrix((np.maximum(w_[first], 1e-9), (u_[first], v_[first])), shape=(n, n))
    return m, idx


def _path_edges(pred_row, src, dst, idx):
    """Edge indices on the shortest path src -> dst from a scipy predecessor row (or None if unreachable)."""
    if src == dst:
        return []
    out, cur = [], dst
    while cur != src:
        p = pred_row[cur]
        if p < 0:
            return None
        out.append(idx[(p, cur)])
        cur = p
    return out[::-1]


def assign_cars(n_nodes, eu, ev, t0, cap, trips, steps=12, alpha=0.15, power=4.0):
    """Time-stepped incremental assignment.
    eu, ev, t0 (s), cap (veh/h): edge arrays. trips: DataFrame with o, d (node index), weight (veh/h represented), step (0..steps-1).
    Returns dict: volume (veh/h per edge), time (final congested s per edge), trip_time (s per trip at its departure step, NaN if
    unreachable), paths (list of edge-index lists)."""
    eu, ev = np.asarray(eu), np.asarray(ev)
    vol = np.zeros(len(eu))
    tt = np.asarray(t0, float).copy()
    trip_time = np.full(len(trips), np.nan)
    paths = [None] * len(trips)
    for k in range(steps):
        sel = np.flatnonzero(trips.step.to_numpy() == k)
        if len(sel) == 0:
            continue
        m, idx = _csr(n_nodes, eu, ev, tt)
        origins = np.unique(trips.o.to_numpy()[sel])
        dist, pred = _sp_dijkstra(m, directed=True, indices=origins, return_predecessors=True)
        row = {o: i for i, o in enumerate(origins)}
        for j in sel:
            o, d, w = int(trips.o.iat[j]), int(trips.d.iat[j]), float(trips.weight.iat[j])
            r = row[o]
            if not np.isfinite(dist[r, d]):
                continue
            p = _path_edges(pred[r], o, d, idx)
            if p is None:
                continue
            paths[j] = p
            trip_time[j] = dist[r, d]
            vol[p] += w
        # hourly-rate estimate of the volume loaded so far, then BPR
        tt = bpr(t0, vol * steps / (k + 1), cap, alpha, power)
    return {"volume": vol, "time": tt, "trip_time": trip_time, "paths": paths}


def crossing_wait(flow_vph, signalised, crossing_m, cycle_s=90.0, ped_green_s=25.0, walk_speed=1.2, start_up_s=2.0, cap_s=90.0):
    """Mean pedestrian wait (s) to cross at a junction.
    Signalised: uniform arrival on red, wait = (C - G)^2 / (2C), charged only where cars flow.
    Unsignalised: gap acceptance (HCM): critical gap t_c = crossing / speed + start-up; wait = (e^(q t_c) - q t_c - 1) / q, q in veh/s."""
    q = np.asarray(flow_vph, float) / 3600.0
    tc = np.asarray(crossing_m, float) / walk_speed + start_up_s
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gap = np.where(q > 0, (np.exp(np.minimum(q * tc, 50)) - q * tc - 1) / np.where(q > 0, q, 1), 0.0)
    sig = np.where(q > 0, (cycle_s - ped_green_s) ** 2 / (2 * cycle_s), 0.0)
    return np.minimum(np.where(np.asarray(signalised, bool), sig, gap), cap_s)


def walk_to_nearest(n_nodes, eu, ev, cost, length, extra_attrs, origins, dests):
    """Pedestrian routing: each origin node to its cheapest destination node (multi-source search on the reversed graph).
    eu/ev/cost: directed edges (include both directions). extra_attrs: dict name -> per-edge array, summed along each route.
    Returns DataFrame (one row per origin) with cost, length_m, reached, <attr sums>, and the list of edge paths."""
    m, idx = _csr(n_nodes, np.asarray(ev), np.asarray(eu), np.asarray(cost, float))    # reversed graph
    dests = np.unique(np.asarray(dests, int))
    dist, pred, src = _sp_dijkstra(m, directed=True, indices=dests, min_only=True, return_predecessors=True)
    rows, paths = [], []
    length = np.asarray(length, float)
    for o in np.asarray(origins, int):
        if not np.isfinite(dist[o]):
            rows.append({"cost": np.nan, "length_m": np.nan, "reached": False, **{k: np.nan for k in extra_attrs}})
            paths.append(None)
            continue
        p, cur = [], o
        while pred[cur] >= 0:            # walk back towards the destination; reversed edge (pred -> cur) is original (cur -> pred)
            nxt = pred[cur]
            p.append(idx[(nxt, cur)])
            cur = nxt
        p = np.array(p, int)
        rows.append({"cost": float(dist[o]), "length_m": float(length[p].sum()) if len(p) else 0.0, "reached": True,
                     **{k: float(np.asarray(a, float)[p].sum()) if len(p) else 0.0 for k, a in extra_attrs.items()}})
        paths.append(p)
    return pd.DataFrame(rows), paths


def heat_exposure(shade, lst_c, lst_lo=35.0, lst_hi=55.0, w_sun=0.5):
    """Per-edge heat exposure in [0, 1]: w_sun * (1 - shade) + (1 - w_sun) * normalised surface temperature."""
    s = np.clip(np.asarray(shade, float), 0, 1)
    t = np.clip((np.asarray(lst_c, float) - lst_lo) / (lst_hi - lst_lo), 0, 1)
    return w_sun * (1 - s) + (1 - w_sun) * np.nan_to_num(t, nan=0.5)
