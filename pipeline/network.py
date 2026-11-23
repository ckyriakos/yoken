"""Walking network from OSM street lines, and population-weighted accessibility to destinations."""
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.ops import unary_union, linemerge  # noqa: F401


def build_graph(lines, exclude=()):
    """lines: GeoDataFrame (metric CRS) with a 'highway' column. Lines are NODED at every crossing (OSM exports often keep ways whole),
    so two ways that cross become connected. This also joins bridges/underpasses that cross without connecting; it is a known simplification.
    Returns (graph undirected, QA dict). Node ids are integers; coordinates in graph.graph['xy']."""
    ln = lines[~lines.highway.isin(exclude)]
    ln = ln[ln.geometry.notna() & ~ln.geometry.is_empty]
    geoms = [g for g in ln.explode(index_parts=False).geometry if g.geom_type == "LineString" and g.length > 0]
    noded = unary_union(geoms)
    segs = [noded] if noded.geom_type == "LineString" else list(noded.geoms)
    key, xy, G = {}, [], nx.Graph()

    def nid(c):
        k = (round(c[0], 1), round(c[1], 1))
        if k not in key:
            key[k] = len(xy); xy.append(k); G.add_node(key[k])
        return key[k]
    for s in segs:
        c = list(s.coords)
        a, b = nid(c[0]), nid(c[-1])
        if a != b:
            L = s.length
            if not G.has_edge(a, b) or G[a][b]["length"] > L:
                G.add_edge(a, b, length=L, geom=s)
    G.graph["xy"] = np.array(xy)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    qa = {"input_lines": int(len(ln)), "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "components": len(comps),
          "largest_component_node_share": round(len(comps[0]) / G.number_of_nodes(), 4) if comps else 0.0,
          "network_km": round(sum(d["length"] for *_, d in G.edges(data=True)) / 1000, 1)}
    return G, qa


def snap(G, pts_xy, max_m):
    """Nearest graph node (largest component only) for each point. Returns (node ids, distance m, ok mask within max_m)."""
    comps = max(nx.connected_components(G), key=len)
    ids = np.array(sorted(comps))
    tree = cKDTree(G.graph["xy"][ids])
    d, i = tree.query(np.asarray(pts_xy))
    return ids[i], d, d <= max_m


def dist_to_nearest(G, dest_nodes, cutoff_m):
    """Walking distance (m) from every node to the nearest destination node; nodes beyond cutoff are absent."""
    dest = [n for n in set(dest_nodes) if n in G]
    if not dest:
        return {}
    return nx.multi_source_dijkstra_path_length(G, dest, cutoff=cutoff_m, weight="length")


def summarise_access(origin_nodes, origin_snap_m, pop, nbid, dist, cutoff_m, speed, minutes=(5, 10)):
    """Population-weighted accessibility by neighbourhood. Distance = snap distance + network distance. Beyond cutoff counts as unreachable
    (capped at cutoff for the mean, so means are floors for poorly served areas)."""
    d = np.array([dist.get(n, np.inf) for n in origin_nodes]) + np.asarray(origin_snap_m)
    reach = d <= cutoff_m
    capped = np.where(reach, d, cutoff_m)
    df = pd.DataFrame({"nbid": nbid, "pop": pop, "min": capped / speed / 60, "reach": reach})
    for m in minutes:
        df[f"w{m}"] = d <= m * 60 * speed
    g = df.assign(**{c: df[c] * df["pop"] for c in ["min", "reach"] + [f"w{m}" for m in minutes]}).groupby("nbid")
    P = g["pop"].sum().replace(0, np.nan)
    out = pd.DataFrame({"mean_min": g["min"].sum() / P, "unreachable_pct": 100 * (1 - g["reach"].sum() / P)})
    for m in minutes:
        out[f"within{m}_pct"] = 100 * g[f"w{m}"].sum() / P
    return out
