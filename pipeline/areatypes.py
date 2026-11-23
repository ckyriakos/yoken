"""Rule-based area types and candidate actions. Rules live in area_types.yaml and actions.yaml."""
import numpy as np
import pandas as pd

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b}


def derive(df):
    d = df.copy()
    d["lu_industrial_special_pct"] = d.lu_industrial_craft_pct + d.lu_special_zone_pct
    d["lu_centre_pct"] = d.lu_central_commercial_pct + d.lu_local_centre_pct
    return d


def tiers(d, elig):
    """Tertile tiers of P90 LST and NDVI among eligible neighbourhoods (same logic as the app's profile labels)."""
    e = d[elig]
    hq = e.lst_p90_c.quantile([1 / 3, 2 / 3]).to_numpy()
    vq = e.ndvi_mean.quantile([1 / 3, 2 / 3]).to_numpy()
    d = d.copy()
    d["heat_tier"] = np.select([~elig, d.lst_p90_c >= hq[1], d.lst_p90_c <= hq[0]], ["n/a", "high", "low"], default="mid")
    d["veg_tier"] = np.select([~elig, d.ndvi_mean >= vq[1], d.ndvi_mean <= vq[0]], ["n/a", "high", "low"], default="mid")
    return d


def _threshold(d, c):
    if "q" in c:
        return float(d[c["col"]].dropna().quantile(c["q"]))
    return c["value"]


def _holds(d, c, thr=None):
    col = d[c["col"]]
    if c["op"] == "in":
        return col.isin(c["value"])
    t = thr if thr is not None else _threshold(d, c)
    return OPS[c["op"]](col, t).fillna(False)


def assign_types(d, rules):
    """First matching rule wins. Quantile thresholds are computed once over the whole table. Returns rule id, type, label, why."""
    thr = {(r["id"], i): _threshold(d, c) for r in rules for i, c in enumerate(r["when"]) if "q" in c}
    out = pd.DataFrame(index=d.index, columns=["area_type", "area_type_label", "area_type_rule", "area_type_why"], dtype=object)
    open_ = pd.Series(True, index=d.index)
    for r in rules:
        m = open_.copy()
        for i, c in enumerate(r["when"]):
            m &= _holds(d, c, thr.get((r["id"], i)))
        out.loc[m, ["area_type", "area_type_label", "area_type_rule", "area_type_why"]] = [r["type"], r["label"], r["id"], r["why"]]
        open_ &= ~m
    return out, {f"{k[0]}#{k[1]}": v for k, v in thr.items()}


def assign_actions(d, actions, eligible):
    """Long table: one row per (neighbourhood, candidate action). Ineligible neighbourhoods get none (they are not ranked)."""
    rows = []
    for a in actions:
        m = eligible.copy()
        for c in a["when"]:
            m &= _holds(d, c)
        for nid in d.loc[m, "neighbourhood_id"]:
            rows.append({"neighbourhood_id": nid, "action_id": a["id"], "action": a["label"], "mechanism": a["mechanism"],
                         "abm_lever": a["abm"]["lever"], "abm_metrics": ";".join(a["abm"]["metrics"]),
                         "side_effects": ";".join(a["abm"]["side_effects"])})
    return pd.DataFrame(rows, columns=["neighbourhood_id", "action_id", "action", "mechanism", "abm_lever", "abm_metrics", "side_effects"])
