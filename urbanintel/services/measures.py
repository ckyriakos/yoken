"""Neighbourhood area types and the cooling measures that fit them (from the heat pipeline: build_area_types.py, actions.yaml,
decision_map.yaml). Measures are OPTIONS TO EVALUATE, not recommendations; each carries whether this platform can simulate it:
  tested        a simulation on the Heat page estimates it directly (street trees, cooling refuges)
  proxy         something close is simulated (pocket parks ~ new cooling destination; de-paving ~ land-cover what-if)
  not_testable  data or a model is missing (the reason is shown)"""
from __future__ import annotations

import html
import json
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import yaml

from .. import layers as LY

OUT = LY.OUT / "demo" / "area_types.geojson"
STATUS = {"tested": ("Simulated here", "#1b7f3b", "●"), "proxy": ("Approximated", "#b54708", "◐"),
          "not_testable": ("Not testable yet", "#5b6573", "○"), "none": ("Check first", "#5b6573", "○")}
TYPE_COLOURS = {"Dense residential": "#b42318", "Residential": "#f79009", "Mixed use": "#fdb022", "Commercial / local centre": "#7a5af8",
                "Civic / institutional": "#2e90fa", "Industrial / logistics (built form confirms)": "#475467", "Green / open space": "#1b7f3b",
                "Unclassified zoning": "#98a2b3", "Insufficient zoning data": "#d0d5dd"}
TIER = {"high": "high", "mid": "medium", "low": "low"}


@lru_cache(maxsize=1)
def _rules():
    acts = {a["id"]: a for a in yaml.safe_load(open(LY.ROOT / "actions.yaml"))["actions"]}
    dm = yaml.safe_load(open(LY.ROOT / "decision_map.yaml"))["actions"]
    return acts, dm


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    t = pd.read_csv(LY.OUT / "neighbourhood_area_types.csv")
    a = pd.read_csv(LY.OUT / "neighbourhood_actions.csv")
    acts, dm = _rules()
    a["status"] = a.action_id.map(lambda i: dm.get(i, {}).get("status", "none"))
    a["note"] = a.action_id.map(lambda i: dm.get(i, {}).get("note") or dm.get(i, {}).get("reason", ""))
    a["caveat"] = a.action_id.map(lambda i: "; ".join(acts.get(i, {}).get("caveats", [])[:1]))
    order = {"tested": 0, "proxy": 1, "not_testable": 2, "none": 3}
    a = a.sort_values(["neighbourhood_id", "status"], key=lambda s: s.map(order) if s.name == "status" else s)
    m = a.groupby("neighbourhood_id").apply(lambda g: g[["action_id", "action", "mechanism", "status", "note", "caveat"]].to_dict("records"),
                                            include_groups=False)
    t["measures"] = t.neighbourhood_id.map(m).apply(lambda v: v if isinstance(v, list) else [])
    return t


def measures_html(ms, limit=6):
    """Compact list for the info box: status dot, measure, mechanism."""
    if not ms:
        return "<div style='margin-top:4px;color:#5b6573'>No candidate measures (not eligible or no rule matched).</div>"
    out = ["<div style='margin-top:6px;font-weight:700;color:#1f2a37'>Measures that fit this area</div>"]
    for x in ms[:limit]:
        lab, col, dot = STATUS.get(x["status"], STATUS["none"])
        out.append(f"<div style='margin:3px 0'><span style='color:{col}'>{dot}</span> <b>{html.escape(x['action'])}</b>"
                   f"<br><span style='color:#5b6573'>{html.escape(x['mechanism'])} · <i style='color:{col}'>{lab}</i></span></div>")
    out.append("<div style='color:#5b6573;font-size:11px;margin-top:3px'>Options to evaluate, not recommendations. "
               "● simulated on the Heat page · ◐ approximated · ○ needs data first</div>")
    return "".join(out)


def build_layer() -> dict:
    t = table()
    ind = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")[["neighbourhood_id", "neighbourhood_name_en", "lst_p90_c", "ndvi_mean",
                                                                      "geometry"]]
    g = ind.merge(t[["neighbourhood_id", "area_type_label", "area_type_why", "heat_tier", "veg_tier", "eligible", "priority_score",
                     "measures"]], on="neighbourhood_id", how="left")
    g["heat"] = g.heat_tier.map(TIER)
    g["greenery"] = g.veg_tier.map(TIER)
    g["measures_html"] = g.measures.apply(measures_html)
    g["n_measures"] = g.measures.apply(len)
    g = g.drop(columns=["measures", "heat_tier", "veg_tier"])
    g["geometry"] = g.geometry.simplify(0.00003)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(OUT, driver="GeoJSON")
    return json.loads(g.to_json(drop_id=True))


def for_neighbourhoods(ids) -> list[dict]:
    t = table().set_index("neighbourhood_id")
    out = []
    for i in ids:
        if i in t.index:
            r = t.loc[i]
            out.append({"neighbourhood_id": i, "name": r.neighbourhood_name, "area_type": r.area_type_label, "why": r.area_type_why,
                        "heat": TIER.get(r.heat_tier, r.heat_tier), "greenery": TIER.get(r.veg_tier, r.veg_tier), "measures": r.measures})
    return out
