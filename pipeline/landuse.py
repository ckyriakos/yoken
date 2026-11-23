"""Zoned land use -> broad area types -> per-neighbourhood shares."""
import geopandas as gpd
import numpy as np
import pandas as pd
from .crosswalk import norm


def classify(labels, rules, default="other"):
    """First matching rule wins. Matching uses crosswalk.norm (case, accents, Latin/Greek look-alikes)."""
    comp = [(r["class"], [norm(k) for k in r["any"]]) for r in rules]
    def one(s):
        n = norm(s) if isinstance(s, str) else ""
        return next((c for c, kws in comp if any(k in n for k in kws)), default)
    m = {s: one(s) for s in labels.dropna().unique()}
    return labels.map(m).fillna(default)


def classify_landuse(lu, rules, primary="ypomnima", fallback="xriseis"):
    """Try the legend label first; where that gives 'other', try the use label."""
    a = classify(lu[primary], rules)
    b = classify(lu[fallback], rules)
    return a.where(a != "other", b)


def neighbourhood_shares(lu, nb, class_keys):
    """lu has landuse_class, nb has neighbourhood_id; both in a metric CRS. Shares are % of the neighbourhood's ZONED area."""
    x = gpd.overlay(lu[["landuse_class", "geometry"]], nb[["neighbourhood_id", "geometry"]],
                    how="intersection", keep_geom_type=True)
    x["a"] = x.area
    t = (x.pivot_table(index="neighbourhood_id", columns="landuse_class", values="a", aggfunc="sum", fill_value=0.0)
         .reindex(columns=class_keys, fill_value=0.0))
    res = nb[["neighbourhood_id"]].assign(nb_area_m2=nb.area.to_numpy()).merge(t.reset_index(), on="neighbourhood_id", how="left")
    res[class_keys] = res[class_keys].fillna(0.0)
    zoned = res[class_keys].sum(axis=1)
    res["zoned_pct_of_area"] = 100 * zoned / res.nb_area_m2
    p = res[class_keys].div(zoned.replace(0, np.nan), axis=0)
    out = res[["neighbourhood_id", "nb_area_m2", "zoned_pct_of_area"]].copy()
    for c in class_keys:
        out[f"lu_{c}_pct"] = 100 * p[c]
    out["lu_dominant"] = p.idxmax(axis=1)
    out["lu_dominant_pct"] = 100 * p.max(axis=1)
    out["lu_mix_index"] = 1 - (p ** 2).sum(axis=1)          # 0 = single type, higher = more mixed
    return out
