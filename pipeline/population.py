"""Dasymetric population: allocate a known municipal total to buildings in proportion to weight x volume.
This is a MODEL, not a census at building level. Weights per zoned class are assumptions (config population.weights)."""
import geopandas as gpd
import numpy as np
import pandas as pd


def tag_buildings(b, lu, nb):
    """Representative point of each building -> landuse_class (or 'unzoned') and neighbourhood_id. Overlaps: first polygon wins.
    Buildings outside every neighbourhood are dropped (reported by the caller)."""
    p = b[["fp_m2", "Height", "geometry"]].copy()
    p["geometry"] = b.geometry.representative_point()
    j = gpd.sjoin(p, nb[["neighbourhood_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()].drop(columns="index_right")
    k = gpd.sjoin(j, lu[["landuse_class", "geometry"]], how="left", predicate="within")
    k = k[~k.index.duplicated()].drop(columns="index_right")
    k["landuse_class"] = k.landuse_class.fillna("unzoned")
    return k


def impute_height(t):
    """Height 0/missing -> median height (>0) of the same land-use class, else of all buildings. Returns (height, imputed flag)."""
    h = t.Height.where(t.Height > 0)
    med = h.groupby(t.landuse_class).transform("median")
    return h.fillna(med).fillna(h.median()), h.isna()


def allocate(t, total, weights):
    """Population per building = total * w*V / sum(w*V). Classes missing from `weights` get weight 0."""
    h, imputed = impute_height(t)
    out = t.copy()
    out["height_used_m"], out["height_imputed"] = h, imputed
    out["volume_m3"] = out.fp_m2 * h
    raw = out.landuse_class.map(weights).fillna(0.0) * out.volume_m3
    if raw.sum() <= 0:
        raise ValueError("all allocation weights are zero; check population.weights and the land-use classes")
    out["population"] = total * raw / raw.sum()
    return out


def sensitivity_weights(weights, partial_value):
    """Replace every partial weight (0 < w < 1) by partial_value; full (1) and zero weights stay."""
    return {k: (partial_value if 0 < v < 1 else v) for k, v in weights.items()}


def grid_cells(path, crs_out=2100):
    """Valid cells of a census grid raster (e.g. Eurostat 1 km, EPSG:3035) as polygons in crs_out with column grid_pop."""
    import rasterio
    from shapely.geometry import box
    with rasterio.open(path) as r:
        a = r.read(1, masked=True)
        rows, cols = np.where(~a.mask)
        T = r.transform
        geoms = [box(T.c + c * T.a, T.f + (rr + 1) * T.e, T.c + (c + 1) * T.a, T.f + rr * T.e) for rr, c in zip(rows, cols)]   # north-up grid
        g = gpd.GeoDataFrame({"grid_pop": a[rows, cols].astype(float).data}, geometry=geoms, crs=r.crs)
    return g.to_crs(crs_out).reset_index(drop=True)


def compare_to_grid(bp_xy, bp_pop, nb_union, cells, min_cover=0.8):
    """Model population per grid cell vs the grid. bp_xy: (n,2) building coordinates in the cells' CRS; nb_union: study-area polygon.
    Returns a DataFrame with grid_pop, model_pop, cover (share of the cell inside the study area) and ratio = model/grid (only where cover >= min_cover)."""
    c = cells.copy()
    c["cover"] = c.geometry.intersection(nb_union).area / c.geometry.area
    pts = gpd.GeoDataFrame({"pop": np.asarray(bp_pop, float)}, geometry=gpd.points_from_xy(bp_xy[:, 0], bp_xy[:, 1]), crs=cells.crs)
    j = gpd.sjoin(pts, c[["geometry"]], how="inner", predicate="within")
    c["model_pop"] = j.groupby("index_right")["pop"].sum().reindex(c.index).fillna(0.0)
    c = c[c.cover > 0].drop(columns="geometry").copy()
    c["ratio"] = np.where(c.cover >= min_cover, c.model_pop / c.grid_pop, np.nan)
    return c


def grid_cells_all(path, study_poly, crs_poly=2100):
    """Every grid cell overlapping study_poly's bounding box, in crs_poly, with grid_pop. Cells absent from the raster (nodata) get 0:
    GEOSTAT omits cells with too few people, so inside the study area nodata means 'about none' (0-2 people)."""
    import rasterio
    from shapely.geometry import box
    with rasterio.open(path) as r:
        a = r.read(1, masked=True)
        T = r.transform
        pr = gpd.GeoSeries([study_poly], crs=crs_poly).to_crs(r.crs).iloc[0].bounds
        c0, c1 = int(np.floor((pr[0] - T.c) / T.a)), int(np.ceil((pr[2] - T.c) / T.a))
        r0, r1 = int(np.floor((pr[3] - T.f) / T.e)), int(np.ceil((pr[1] - T.f) / T.e))
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, r.width), min(r1, r.height)
        rows = [(rr, c) for rr in range(r0, r1) for c in range(c0, c1)]
        geoms = [box(T.c + c * T.a, T.f + (rr + 1) * T.e, T.c + (c + 1) * T.a, T.f + rr * T.e) for rr, c in rows]
        vals = [0.0 if a.mask[rr, c] else float(a[rr, c]) for rr, c in rows]
        g = gpd.GeoDataFrame({"grid_pop": vals}, geometry=geoms, crs=r.crs)
    return g.to_crs(crs_poly).reset_index(drop=True)


def calibrate(pop, cell, target, total):
    """Rescale building populations so that each controlled grid cell sums to its target, and everything else absorbs the remainder.
    pop: per-building array; cell: per-building cell id (NaN if outside every cell); target: Series cell id -> people (controlled cells only).
    Returns (new pop array, info dict). Within a cell the allocation pattern is unchanged: only the cell total is controlled."""
    s = pd.Series(np.asarray(pop, float))
    cell = pd.Series(np.asarray(cell, float))
    ctrl = cell.isin(target.index)
    model_c = s[ctrl].groupby(cell[ctrl]).sum()
    ok = model_c[model_c > 0].index
    factor = (target.reindex(ok) / model_c.reindex(ok))
    new = s.copy()
    new[ctrl] = s[ctrl] * cell[ctrl].map(factor).fillna(0.0)
    unplaced = float(target.drop(ok, errors="ignore").sum())               # controlled people in cells with no modelled buildings
    remainder = float(total - new[ctrl].sum())
    u = float(s[~ctrl].sum())
    if remainder <= 0 or u <= 0:
        raise ValueError(f"cannot absorb the remainder: remainder {remainder:.0f}, uncontrolled model population {u:.0f}")
    new[~ctrl] = s[~ctrl] * remainder / u
    return new.to_numpy(), {"controlled_cells": int(len(ok)), "factor": factor, "remainder": remainder, "uncontrolled_scale": remainder / u,
                            "unplaced_people": unplaced}
