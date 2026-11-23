"""Neighbourhood building context from the solar-map footprints (columns: Height, Area, USAB_AREA, ...)."""
import geopandas as gpd
import pandas as pd


def clean(b):
    """Drop empty geometries, repair invalid ones, and return (buildings, data-quality dict). b must be in a metric CRS."""
    qa = {"rows": int(len(b))}
    b = b[b.geometry.notna() & ~b.geometry.is_empty].copy()
    qa["invalid_geometries"] = int((~b.geometry.is_valid).sum())
    b["geometry"] = b.geometry.make_valid()
    b = b[b.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    b["fp_m2"] = b.area
    qa["height_zero_share"] = round(float((b.Height <= 0).mean()), 4)
    qa["footprint_lt_10m2_share"] = round(float((b.fp_m2 < 10).mean()), 4)
    qa["height_p50_p99_max_m"] = [float(b.Height.quantile(.5)), float(b.Height.quantile(.99)), float(b.Height.max())]
    qa["footprint_area_km2"] = round(float(b.fp_m2.sum() / 1e6), 3)
    return b, qa


def summarise(b, nb, large_m2=500.0):
    """Per-neighbourhood context. b (from clean) and nb (needs neighbourhood_id) share a metric CRS.
    Buildings are assigned by representative point, so footprints straddling a boundary count wholly to one side."""
    pts = b.copy()
    pts["geometry"] = b.geometry.representative_point()
    j = gpd.sjoin(pts, nb[["neighbourhood_id", "geometry"]], how="inner", predicate="within")
    j = j[~j.index.duplicated()]
    j["h"] = j.Height.where(j.Height > 0)
    j["hxa"] = j.h * j.fp_m2
    j["big"] = j.fp_m2 >= large_m2
    j["big_m2"] = j.fp_m2.where(j.big, 0.0)
    g = j.groupby("neighbourhood_id")
    out = pd.DataFrame({
        "n_buildings": g.size(), "footprint_m2": g.fp_m2.sum(),
        "n_height_missing": g.apply(lambda d: int(d.h.isna().sum()), include_groups=False),
        "h_area_weighted_m": g.hxa.sum() / g.apply(lambda d: d.fp_m2[d.h.notna()].sum(), include_groups=False),
        "h_max_m": g.h.max(), "volume_m3": g.hxa.sum(),
        "n_large_roofs": g.big.sum(), "large_roof_m2": g.big_m2.sum(),
        "pv_usable_roof_m2": g.USAB_AREA.sum() if "USAB_AREA" in j else float("nan")}).reset_index()
    res = nb[["neighbourhood_id"]].assign(nb_area_m2=nb.area.to_numpy()).merge(out, on="neighbourhood_id", how="left")
    for c in ["n_buildings", "footprint_m2", "n_height_missing", "volume_m3", "n_large_roofs", "large_roof_m2"]:
        res[c] = res[c].fillna(0)
    res["building_cover_pct"] = 100 * res.footprint_m2 / res.nb_area_m2
    res["mean_height_all_area_m"] = res.volume_m3 / res.nb_area_m2          # volume per ground area
    res["large_roof_cover_pct"] = 100 * res.large_roof_m2 / res.nb_area_m2
    res["unassigned_buildings"] = len(b) - len(j)
    return res
