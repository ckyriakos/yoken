"""Destination node sets on the walking graph (same definitions as build_access.py)."""
import geopandas as gpd
import numpy as np
from shapely.ops import unary_union
from .crosswalk import norm


def nodes_gdf(G, crs=2100):
    xy = G.graph["xy"]
    g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs)
    g["nid"] = np.arange(len(g))
    return g


def _entry_nodes(nodes, geoms, entry_m):
    poly = gpd.GeoDataFrame(geometry=gpd.GeoSeries(list(geoms), crs=nodes.crs).buffer(entry_m).reset_index(drop=True), crs=nodes.crs)
    return gpd.sjoin(nodes, poly, how="inner", predicate="within").nid.unique()


def green_entry_nodes(lu, nodes, min_m2=5000, entry_m=25):
    """Nodes within entry_m of green/open zoned land after merging touching pieces and keeping parcels of at least min_m2."""
    g = unary_union(lu.loc[lu.landuse_class == "green_open", "geometry"].tolist())
    parts = [p for p in (g.geoms if hasattr(g, "geoms") else [g]) if p.area >= min_m2]
    return _entry_nodes(nodes, parts, entry_m)


def refuge_entry_nodes(lu, nodes, skip=("ΕΚΠΑΙΔΕΥΣ", "ΣΧΟΛΕΙΟΥ", "ΒΡΕΦΟΝΗΠΙΑΚ", "ΜΑΖΙΚΩΝ ΜΕΤΑΦΟΡΩΝ", "ΑΘΛΗΤΙΣΜΟΣ"), entry_m=25):
    """Nodes within entry_m of civic-zoned land, excluding education, sport, transit and nursery labels (placeholder for real refuges)."""
    skip = [norm(k) for k in skip]
    lab = (lu.ypomnima.fillna("") + " " + lu.xriseis.fillna("")).map(norm)
    civ = lu[(lu.landuse_class == "civic") & ~lab.map(lambda t: any(k in t for k in skip))]
    return _entry_nodes(nodes, civ.geometry, entry_m)
