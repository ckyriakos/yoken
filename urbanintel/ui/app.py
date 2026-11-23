"""Athens Urban Intelligence: demo UI (Solara + leafmap).

Run from the project root:   solara run urbanintel/ui/app.py     (http://localhost:8765)
Pages: Overview (all GIS layers) | Kallimarmaro (Vas. Olgas ABM) | Heat (proposal -> ABM) | Flood (zones, network, Sentinel-2).
All data comes from urbanintel.services (the same code behind the API); run `python -m urbanintel.prepare --cases` once first."""
from __future__ import annotations

import datetime
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))     # project root, so `solara run urbanintel/ui/app.py` works

import ipyleaflet as L
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import solara

from urbanintel import layers as LY
from urbanintel.services import flood, heat, landcover, measures, mobility, overview, timelapse, traffic
from urbanintel.ui import maplayers as ML

MAP_H = "calc(100vh - 70px)"
CSS = """
.ui-side {width: 330px; min-width: 330px; height: calc(100vh - 70px); overflow-y: auto; padding: 4px 14px 14px 4px;}
.ui-main {flex: 1; min-width: 0;}
.kpis {display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 6px 0 12px;}
.kpi {background: #f6f7f9; border-radius: 8px; padding: 8px 10px; border: 1px solid #e6e8eb;}
.kpi .v {font-size: 20px; font-weight: 600; color: #1f2a37; line-height: 1.2;}
.kpi .l {font-size: 11.5px; color: #5b6573;}
.kpi .d {font-size: 12px; font-weight: 600;}
.good {color: #1b7f3b;} .bad {color: #b42318;} .neutral {color: #5b6573;}
.grp {font-size: 12px; font-weight: 700; color: #1f2a37; text-transform: uppercase; letter-spacing: .04em; margin: 12px 0 2px;}
.note {font-size: 12px; color: #5b6573; line-height: 1.4;}
.chat {margin-top: 16px; padding: 10px; border: 1px dashed #b8c0cc; border-radius: 8px; font-size: 12.5px; color: #5b6573;}
.legend {font-size: 12px; margin-top: 4px;} .legend span.sw {display:inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: -2px;}
h3.title {margin: 2px 0 2px; font-size: 18px; color: #1f2a37;}
.ui-side .v-input {--v-input-control-height: 26px; --v-input-padding-top: 0; min-height: 0 !important; margin: 0 !important;}
.ui-side .v-input__details {display: none !important;}
.ui-side .v-selection-control {min-height: 26px !important;}
.ui-side .v-selection-control__wrapper, .ui-side .v-selection-control__input {height: 26px !important; width: 26px !important;}
.ui-side .v-checkbox label {font-size: 13.5px;}
.ui-side .v-btn-toggle, .ui-side .v-btn-group {flex-wrap: wrap; height: auto !important;}
.ui-side .v-btn-group .v-btn {height: 28px !important;}
.ui-side .v-btn {text-transform: none; letter-spacing: 0;}
.solara-loader, #loader, .v-application__wrap > div > div[style*="bottom: 0px"] {display: none !important;}
"""


# ------------------------------------------------------------------------------------------------------------ small parts
def fmt(x, nd=1, unit="", sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}"
    return s + unit


@solara.component
def Kpi(label, value, delta=None, good_if="down", ref="vs today"):
    cls = "neutral"
    num = delta
    if isinstance(delta, str):
        try:
            num = float(delta.replace(",", "").split()[0])
        except ValueError:
            num = None
    if isinstance(num, (int, float)) and abs(num) > 1e-9:
        cls = "good" if (num < 0) == (good_if == "down") else "bad"
    d = "" if delta is None else (delta if isinstance(delta, str) else fmt(delta, 2, sign=True))
    solara.HTML(tag="div", classes=["kpi"], unsafe_innerHTML=f"<div class='v'>{value}</div><div class='l'>{label}</div>"
                + (f"<div class='d {cls}'>{d} {ref}</div>" if d else ""))


def legend_html(items):
    return "<div class='legend'>" + "<br>".join(f"<span class='sw' style='background:{c}'></span>{l}" for l, c in items) + "</div>"


def replace_layer(m, old, new):
    if old is not None and old in m.layers:
        m.remove_layer(old)
    if new is not None:
        m.add_layer(new)
    return new


NB_NAMES = None


def nb_names():
    global NB_NAMES
    if NB_NAMES is None:
        import geopandas as gpd
        g = gpd.read_file(LY.OUT / "neighbourhood_indicators.geojson")
        NB_NAMES = dict(zip(g.neighbourhood_id, g.neighbourhood_name_en.fillna(g.neighbourhood_name)))
    return NB_NAMES


# ------------------------------------------------------------------------------------------------------------ Overview
CASE_GROUPS = {"Case: Vas. Olgas"}


@solara.component
def Overview(districts=()):
    m = solara.use_memo(lambda: ML.new_map(height=MAP_H, fit_municipality=True), [])
    solara.use_effect(lambda: ML.set_focus(m, districts), [districts])
    built = solara.use_memo(dict, [])           # layer id -> widget (built once, then only shown / hidden)
    flood.register_layers()
    defaults = {l.id for l in LY.LAYERS if l.default_on}
    on, set_on = solara.use_state(defaults)
    detail, set_detail = solara.use_state("Standard")
    busy, set_busy = solara.use_state(False)

    def show(lid, visible):
        """Idempotent and serialised: safe under fast clicking."""
        with m._lock:
            w = built.get(lid)
            if w is None and visible:
                w = built[lid] = ML.build_layer(lid, m, detail)
                m.add_layer(w)
            elif w is not None:
                if visible and w not in m.layers:
                    m.add_layer(w)
                w.visible = visible

    def init():
        for lid in defaults:
            show(lid, True)
    solara.use_effect(init, [])

    def toggle(lid):
        def f(v):
            set_busy(True)
            try:
                show(lid, v)
            finally:
                set_busy(False)
            set_on(lambda prev: (prev | {lid}) if v else (prev - {lid}))
        return f

    def change_detail(d):
        set_busy(True)
        with m._lock:
            for lid, w in list(built.items()):
                if LY.BY_ID[lid].kind in ("polygon", "line"):
                    vis = w.visible
                    if w in m.layers:
                        m.remove_layer(w)
                    nw = built[lid] = ML.build_layer(lid, m, d)
                    nw.visible = vis
                    m.add_layer(nw)
        set_detail(d)
        set_busy(False)

    h = overview.headline(districts)
    with solara.Row(style={"gap": "0", "align-items": "stretch"}):
        with solara.Div(classes=["ui-side"]):
            solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Athens at a glance" if not districts else
                        ", ".join(overview.ORD[d] for d in districts) + (" district" if len(districts) == 1 else " districts"))
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Every layer of the Athens GIS repository plus the heat and flood analyses. "
                        "Click anything on the map (or hover small layers) for an explanation. All layers are also served by the API at <code>/api/v1/layers</code>.")
            with solara.Div(classes=["kpis"]):
                Kpi("residents (modelled, 2021)", fmt(h.get("population"), 0))
                Kpi("hottest-day surface temp. (P90, °C)", fmt(h.get("lst_p90_mean_c"), 1))
                Kpi("residents ≤5 min walk to green", fmt(h.get("green_within5_pct"), 0, "%"))
                Kpi("residents in 100-yr flood zone", fmt(h.get("flood100_pop_pct"), 1, "%"))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Map detail")
            solara.ToggleButtonsSingle(value=detail, on_value=change_detail, values=list(ML.DETAIL_TOL), dense=True)
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Fast simplifies shapes to ~10 m, Standard to ~3 m, Full uses the original "
                        "geometry. Zoom in up to street level (zoom 19); the basemap can be switched at the bottom right of the map.")
            for g in [g for g in dict.fromkeys(l.group for l in LY.LAYERS) if g not in CASE_GROUPS]:
                solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML=g)
                for l in [l for l in LY.LAYERS if l.group == g and l.id not in LY.BBOX_ONLY]:
                    solara.Checkbox(label=l.title, value=l.id in on, on_value=toggle(l.id), disabled=busy)
            active = [l for l in LY.LAYERS if l.id in on and (l.kind in ("raster", "heat") or l.style.get("choropleth")
                                                              or l.style.get("categorical") or l.group == "Sensors")]
            if active:
                solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="What am I looking at?")
                for l in active[::-1]:                      # newest first
                    solara.HTML(tag="div", unsafe_innerHTML=ML.index_card_html(l.id))
            solara.HTML(tag="div", classes=["chat"], unsafe_innerHTML="<b>Ask Athens</b> (coming soon)<br>Natural-language questions such as "
                        "<i>“show flood risk and bus stops around Kolonos”</i> will be answered by an assistant calling the same API.")
        with solara.Div(classes=["ui-main"]):
            solara.display(m)


# ------------------------------------------------------------------------------------------------------------ Kallimarmaro
SCN_COL = {"baseline": "#5b6573", "as_built": "#b54708", "traffic_only": "#b42318", "trees_only": "#1b7f3b"}
NDVI_VIEWS = {"NDVI before (2019)": "olgas_ndvi_pre_works", "NDVI during works (2025)": "olgas_ndvi_works",
              "NDVI after (2026)": "olgas_ndvi_after", "NDVI change": "olgas_ndvi_change"}
STREETS = {"amalias": "Amalias", "konstantinou": "Vas. Konstantinou", "arditou": "Arditou", "all": "all three"}


@solara.component
def Kallimarmaro(districts=()):
    res = solara.use_memo(mobility.result, [])
    edges = solara.use_memo(lambda: ML.round_coords(mobility.layer("edges"), 5), [])
    traces = solara.use_memo(lambda: ML.round_coords(mobility.layer("traces"), 5), [])
    signals = solara.use_memo(lambda: mobility.layer("signals"), [])
    lc = solara.use_memo(landcover_result, [])
    cal = solara.use_memo(traffic.calibration_result, [])
    scn, set_scn = solara.use_state("as_built")
    view, set_view = solara.use_state("Change vs before")
    m = solara.use_memo(lambda: ML.new_map(center=(37.9700, 23.7360), zoom=16, height=MAP_H), [])
    ref = solara.use_ref({"edges": None, "traces": [], "ras": None, "zones": None, "sig": None, "det": None})

    def draw():
        sv, sid = view, scn
        for k in ("ras", "zones"):
            if ref.current[k] is not None and ref.current[k] in m.layers:
                m.remove_layer(ref.current[k])
            ref.current[k] = None
        if sv in NDVI_VIEWS:
            lid = NDVI_VIEWS[sv]
            if lid in LY.BY_ID:
                ref.current["ras"] = ML.build_layer(lid, m)
                ref.current["ras"].opacity = 0.85
                m.add_layer(ref.current["ras"])
            z = landcover.zones()
            if z:
                ref.current["zones"] = L.GeoJSON(data=z, style_callback=lambda f: {
                    "color": "#111" if f["properties"]["zone"] == "corridor" else "#6941c6", "weight": 2, "fill": False,
                    "dashArray": None if f["properties"]["zone"] == "corridor" else "4"})
                m.add_layer(ref.current["zones"])
        if sv == "Car volume":
            col, lo, hi, cmap = f"vol_{sid}", 0, 1500, "magma_r"
        elif sv == "Shade":
            col, lo, hi, cmap = f"shade_{sid}", 0, 1, "YlGn"
        else:
            col, lo, hi, cmap = f"dvol_{sid}", -300, 300, "RdBu_r"

        def style(f):
            p = f["properties"]
            if sv in NDVI_VIEWS:
                return {"color": "#555", "weight": 0.6, "opacity": 0.35}
            v = p.get(col) or 0
            if sv == "Change vs before" and abs(v) < 15:
                return {"color": "#c9ced6", "weight": 1, "opacity": 0.5}
            if sv == "Car volume" and v <= 0:
                return {"color": "#d9dde3", "weight": 0.8, "opacity": 0.4}
            w = 1.5 + 5 * min(1.0, abs(v) / (hi if sv != "Shade" else 1))
            if p.get("olgas") or p.get("uturn"):
                w += 2
            return {"color": ML.colour(cmap, v, lo, hi), "weight": w, "opacity": 0.9}
        egj = L.GeoJSON(data=edges, style_callback=style, name="edges", hover_style={"color": "#000", "weight": 6})
        slab = res["scenarios"][sid]

        def edge_info(feature=None, **kw):
            if not feature:
                return
            p = feature["properties"]
            rows = {"Road type": p.get("highway"), f"Cars per hour, {slab}": p.get(f"vol_{sid}"),
                    "Change vs before (cars/h)": p.get(f"dvol_{sid}"), "Shade share (0–1)": p.get(f"shade_{sid}")}
            if p.get("olgas"):
                rows["Note"] = "Vas. Olgas: car-free 2020–26, reopened April 2026"
            ML.show_feature(m, p.get("name") or "Street", rows, {"what": "Peak-hour car flow simulated by the agents (seed 0).",
                                                               "read": "Cars per hour summed over both directions of this street segment."})
        egj.on_click(edge_info)                                # click only: 3,800 segments, hover would flood the server while panning
        ref.current["edges"] = replace_layer(m, ref.current["edges"], egj)
        works = sid in ("as_built", "traffic_only")
        sig = L.GeoJSON(data=signals, point_style={"radius": 6, "color": "#111", "fillColor": "#fdb022", "fillOpacity": 1, "weight": 2}) \
            if works and signals and sv not in NDVI_VIEWS else None
        if sig is not None:
            ML.bind_info(m, sig, "New traffic signal (2026)", {"what": "Signal added at an end of Vas. Olgas by the redevelopment.",
                                                              "read": "The model adds signal delay for cars and a fixed waiting time for pedestrians here."})
        ref.current["sig"] = replace_layer(m, ref.current["sig"], sig)
        det = None
        if cal and sv in ("Car volume", "Change vs before"):
            fit = {d["id"]: d for k in ("after", "before") if k in cal for d in cal[k]["detectors"]}
            feats = [f for f in cal["stations"]["features"] if f["properties"]["id"] in fit]
            for f in feats:
                d = fit[f["properties"]["id"]]
                f["properties"].update(observed_vph=d["observed_vph"], model_vph=d["model_vph"], ratio=d["ratio"])

            def dstyle(f):
                r = f["properties"].get("ratio") or 1
                return {"radius": 7, "color": "#111", "weight": 1.5, "fillOpacity": 0.95,
                        "fillColor": ML.colour("RdBu_r", float(np.log2(max(min(r, 4), 0.25))), -1, 1)}
            det = L.GeoJSON(data={"type": "FeatureCollection", "features": feats}, point_style={"radius": 7}, style_callback=dstyle)
            ML.bind_info(m, det, "Traffic detector", {"what": "Region of Attica loop detector compared with the model.",
                                                     "read": "Detector ÷ model above 1: the model has too few cars here; below 1: too many."},
                         title_field="id")
        ref.current["det"] = replace_layer(m, ref.current["det"], det)
        for t in ref.current["traces"]:
            if t in m.layers:
                m.remove_layer(t)
        ref.current["traces"] = []
        if sv in NDVI_VIEWS:
            return
        trace_scn = {"trees_only": "baseline"}.get(sid, sid)
        n = {"walk": 0, "car": 0}
        for f in traces["features"]:
            p = f["properties"]
            if p["scenario"] != trace_scn or n[p["mode"]] >= 12:
                continue
            n[p["mode"]] += 1
            locs = [[y, x] for x, y in f["geometry"]["coordinates"]]
            a = L.AntPath(locations=locs, delay=1600 if p["mode"] == "walk" else 700, weight=3, dash_array=[8, 18],
                          color="#1b7f3b" if p["mode"] == "walk" else "#344054", pulse_color="#ffffff", opacity=0.75)
            m.add_layer(a)
            ref.current["traces"].append(a)
    solara.use_effect(draw, [scn, view])
    S = res["summary"]

    def g(k, f="delta", sid=None):
        return (S.get(sid or scn, {}).get(k) or {}).get(f)

    def pct(k):
        base = g(k, "mean", "baseline")
        d = g(k)
        return None if not base or d is None else 100 * d / base

    fc = res.get("forecast_decongestion_pct") or [None, None]
    with solara.Row(style={"gap": "0", "align-items": "stretch"}):
        with solara.Div(classes=["ui-side"]):
            solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Vas. Olgas redevelopment (2026)")
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Reopened in April 2026 as a soft-traffic street: 1 lane per direction for "
                        "cars, buses and tram, ~150 new trees, 100+ luminaires, new signals at both ends, U-turn on Amalias. All-vehicle pilot "
                        "since May 2026; no measured results published yet. "
                        + " ".join(f"<a href='{u}' target='_blank'>[{i + 1}]</a>" for i, u in enumerate(res.get("sources", []))))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Scenario")
            lab2id = {v: k for k, v in res["scenarios"].items()}
            solara.ToggleButtonsSingle(value=res["scenarios"][scn], on_value=lambda v: set_scn(lab2id[v]), values=list(res["scenarios"].values()),
                                       dense=True, style={"flex-wrap": "wrap"})
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Lighter traffic on nearby streets")
            with solara.Div(classes=["kpis"]):
                streets = sorted(((k, lab, pct(f"street_{k}_vph")) for k, lab in STREETS.items() if k != "all"),
                                 key=lambda x: x[2] if x[2] is not None else 0)
                for k, lab, v in streets + [("all", "All three streets", pct("street_all_vph"))]:
                    Kpi(f"{lab}: {fmt(g(f'street_{k}_vph', 'mean', 'baseline'), 0)} → {fmt(g(f'street_{k}_vph', 'mean'), 0)} veh/h",
                        fmt(v, 1, "%", True) if v is not None else "–", None)
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Change in cars per hour at the weekday peak, as drivers switch to the "
                        "reopened Olgas. Agent model calibrated to 22 Attica traffic detectors over 20 weekdays. The municipality's "
                        f"{fc[0]}–{fc[1]}% estimate refers to congestion (delays), so it is not directly comparable.")
            CalibrationPanel(cal)
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Drivers")
            with solara.Div(classes=["kpis"]):
                Kpi("cars on Vas. Olgas (veh/h)", fmt(g("olgas_vph", "mean"), 0))
                Kpi("Pangrati→Syntagma/Plaka trip", fmt(g("car_corridor_trip_min", "mean"), 2, " min"),
                    fmt((g("car_corridor_trip_min") or 0) * 60, 0, " s", True), "down", "vs before")
                Kpi("all car trips (veh·h, peak hour)", fmt(g("car_veh_hours", "mean"), 0), g("car_veh_hours"), "down", "vs before")
                Kpi("U-turn on Amalias (veh/h)", fmt(g("uturn_vph", "mean"), 0))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Residents near Olgas (≤400 m)")
            with solara.Div(classes=["kpis"]):
                Kpi("walking trips along Olgas with car traffic", fmt(g("olgas_walk_trips_with_traffic", "mean"), 0),
                    fmt(g("olgas_walk_trips_with_traffic"), 0, sign=True), "down", "vs before")
                Kpi("sun on the walk to metro (min)", fmt(g("corr_metro_sun_min", "mean"), 2), g("corr_metro_sun_min"), "down", "vs before")
                Kpi("crossing wait to National Garden (s)", fmt((g("corr_national_garden_wait_min", "mean") or 0) * 60, 0),
                    fmt((g("corr_national_garden_wait_min") or 0) * 60, 1, " s", True), "down", "vs before")
                Kpi("walk to metro (min)", fmt(g("corr_metro_time_min", "mean"), 2), g("corr_metro_time_min"), "down", "vs before")
            LandCoverPanel(lc)
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Map")
            views = ["Change vs before", "Car volume", "Shade"] + [v for v, lid in NDVI_VIEWS.items() if lid in LY.BY_ID]
            solara.ToggleButtonsSingle(value=view, on_value=set_view, values=views, dense=True, style={"flex-wrap": "wrap"})
            if view in NDVI_VIEWS:
                lid = NDVI_VIEWS[view]
                lg = ML.legend_items(lid) + [("project corridor", "#111"), ("control ring", "#6941c6")] if lid in LY.BY_ID else []
            else:
                lg = {"Change vs before": [("more cars", "#b2182b"), ("fewer cars", "#2166ac"), ("no change", "#c9ced6")],
                      "Car volume": [("0 veh/h", ML.colour("magma_r", 0, 0, 1500)), ("750", ML.colour("magma_r", 750, 0, 1500)),
                                     ("1500+", ML.colour("magma_r", 1500, 0, 1500))],
                      "Shade": [("sunny", ML.colour("YlGn", 0)), ("half", ML.colour("YlGn", .5)), ("shaded", ML.colour("YlGn", 1))]}[view] + \
                     [("walking agents", "#1b7f3b"), ("car agents", "#344054"), ("new signals", "#fdb022")] + \
                     ([("detector: model too high", ML.colour("RdBu_r", -1, -1, 1)), ("detector: model ≈ count", ML.colour("RdBu_r", 0, -1, 1)),
                       ("detector: model too low", ML.colour("RdBu_r", 1, -1, 1))] if cal and view != "Shade" else [])
            solara.HTML(tag="div", unsafe_innerHTML=legend_html(lg))
            if view in NDVI_VIEWS and NDVI_VIEWS[view] in LY.BY_ID:
                solara.HTML(tag="div", unsafe_innerHTML=ML.index_card_html(NDVI_VIEWS[view]))
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"{res['seeds']} runs × {res['assumptions']['cars']['n_agents']} car agents "
                        f"({res['assumptions']['cars']['peak_trips_vph']:,} veh/h, synthetic) and {res['assumptions']['pedestrians']['n_agents']:,} "
                        "resident agents. Deltas are paired (same agents in every scenario).")
        with solara.Div(classes=["ui-main"]):
            solara.display(m)
    KallimarmaroCharts(res)


@solara.component
def CalibrationPanel(cal):
    solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Calibration: Attica loop detectors")
    if not cal:
        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Not calibrated: run <code>python -m urbanintel.calibrate</code>.")
        return
    k = "after" if "after" in cal else "before"
    c = cal[k]
    with solara.Div(classes=["kpis"]):
        Kpi("observed / model (median)", fmt(c["scale_median"], 2))
        Kpi("error per detector (MAPE)", fmt(c["mape_pct"], 0, "%"))
    lg = cal.get("logged")
    solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"{c['n_detectors']} main-road detectors within 800 m. Fitted to <b>{c['label']}</b>; "
                f"car demand × {cal.get('demand_scale', 1):.2f}. "
                + (f"Live logger: {lg['rows']:,} readings, {lg['from'][:16]} → {lg['to'][:16]}. " if lg else "")
                + "Weekday peaks replace the Saturday fit once the logger has collected them.")


@solara.component
def NeighbourhoodMeasures(scope):
    """Why these neighbourhoods, what kind of place they are and which measures fit (collapsed by default)."""
    items = measures.for_neighbourhoods(list(scope))
    if not items:
        return
    body = []
    for it in items:
        body.append(f"<div style='margin:6px 0 2px'><b>{it['name'].title()}</b> · {it['area_type']} · heat {it['heat']}, greenery "
                    f"{it['greenery']}</div>" + measures.measures_html(it["measures"], limit=4).replace(
                        "<div style='margin-top:6px;font-weight:700;color:#1f2a37'>Measures that fit this area</div>", ""))
    with solara.Details("Why these neighbourhoods, and which measures fit", expand=False):
        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="They have the highest heat priority (hot surfaces, little greenery). "
                    "Trees and cooling refuges are the two measures simulated here." + "".join(body))


def landcover_result():
    try:
        p = landcover.OUT / "landcover.json"
        return landcover.run() if p.exists() else None
    except Exception:
        return None


@solara.component
def LandCoverPanel(lc):
    solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Land cover (Sentinel-2, Jun–Sep)")
    if not lc:
        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Not computed yet: run <code>python -m urbanintel.prepare --cases</code>.")
        return
    P, ch = lc["periods"], (lc["change"] or {}).get("after_vs_pre_works") or {}
    pre, aft = (P.get("pre_works") or {}).get("corridor", {}), (P.get("after") or {}).get("corridor", {})
    d = ch.get("ndvi_mean", {}); t = ch.get("tree_pct", {})
    with solara.Div(classes=["kpis"]):
        Kpi("corridor NDVI 2019 → 2026", f"{fmt(pre.get('ndvi_mean'), 2)} → {fmt(aft.get('ndvi_mean'), 2)}",
            fmt(d.get("did"), 3, sign=True) + "" if d else None, "up", "vs control")
        Kpi("tree-class pixels 2019 → 2026", f"{fmt(pre.get('tree_pct'), 0, '%')} → {fmt(aft.get('tree_pct'), 0, '%')}",
            fmt(t.get("did"), 1, " pp", True) + "" if t else None, "up", "vs control")
    rec = (lc["change"] or {}).get("after_vs_works") or {}
    if rec:
        wk = (P.get("works") or {}).get("corridor", {})
        with solara.Div(classes=["kpis"]):
            Kpi("NDVI since works 2025 → 2026", f"{fmt(wk.get('ndvi_mean'), 2)} → {fmt(aft.get('ndvi_mean'), 2)}",
                fmt(rec.get("ndvi_mean", {}).get("did"), 3, sign=True), "up", "vs control")
            Kpi("green pixels since works (tree + grass)", fmt((rec.get("tree_pct", {}).get("did") or 0) + (rec.get("grass_pct", {}).get("did") or 0), 1, " pp", True),
                None)
    if ch.get("expected_lst_change_c") is not None:
        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"Expected surface-temperature effect of the observed change: "
                    f"<b>{ch['expected_lst_change_c']:+.2f} °C</b> in the corridor (land-cover response model). "
                    f"Corridor {lc['corridor_area_ha']} ha vs a {lc['control_area_ha']} ha control ring imaged by the same scenes. "
                    "Young trees: the canopy signal will grow over the next years.")


@solara.component
def KallimarmaroCharts(res):
    S = res["summary"]
    sc = list(res["scenarios"])
    rows = [("Watched streets flow (veh/h)", "street_all_vph", 1), ("Cars on Olgas (veh/h)", "olgas_vph", 1),
            ("Corridor drive time (s)", "car_corridor_trip_min", 60), ("Walk trips on Olgas with traffic (÷10)", "olgas_walk_trips_with_traffic", 0.1),
            ("Sun on walk to metro (s)", "corr_metro_sun_min", 60)]
    fig = go.Figure()
    for s in sc[1:]:
        fig.add_bar(name=res["scenarios"][s], y=[r[0] for r in rows], x=[(S[s][r[1]]["delta"] or 0) * r[2] for r in rows], orientation="h",
                    marker_color=SCN_COL.get(s, "#888"),
                    error_x=dict(type="data", symmetric=False, array=[((S[s][r[1]]["p95"] or 0) - (S[s][r[1]]["delta"] or 0)) * r[2] for r in rows],
                                 arrayminus=[((S[s][r[1]]["delta"] or 0) - (S[s][r[1]]["p05"] or 0)) * r[2] for r in rows]))
    fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=30, b=10), title="Change vs before (5–95% over runs)",
                      legend=dict(orientation="h", y=-0.15), template="simple_white")
    nb = mobility.by_neighbourhood()
    with solara.Columns([3, 2]):
        solara.FigurePlotly(fig)
        if len(nb):
            b = nb[nb.dest == "metro"].pivot_table(index="neighbourhood_id", columns="scenario", values=["time_min", "sun_min", "wait_min"])
            people = nb[(nb.dest == "metro") & (nb.scenario == "baseline")].set_index("neighbourhood_id").people
            t = pd.DataFrame({"neighbourhood": [nb_names().get(i, i) for i in b.index], "people": people.reindex(b.index).round(0).to_numpy(),
                              "walk to metro Δ (s)": ((b["time_min"]["as_built"] - b["time_min"]["baseline"]) * 60).round(1).to_numpy(),
                              "sun Δ (s)": ((b["sun_min"]["as_built"] - b["sun_min"]["baseline"]) * 60).round(1).to_numpy(),
                              "crossing wait Δ (s)": ((b["wait_min"]["as_built"] - b["wait_min"]["baseline"]) * 60).round(1).to_numpy()})
            with solara.Column():
                solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="<b>As built</b>, by neighbourhood (walk to metro)")
                solara.DataFrame(t.sort_values("people", ascending=False), items_per_page=8)


# ------------------------------------------------------------------------------------------------------------ Heat
@solara.component
def Heat(districts=()):
    scope = solara.use_memo(lambda: tuple(heat.scope_ids(districts=districts)), [districts])
    default_scope = solara.use_memo(lambda: tuple(heat.scope_ids()), [])
    sweep = solara.use_memo(heat.run_case, [])
    trees, set_trees = solara.use_state(1000)
    refs, set_refs = solara.use_state(2)
    dest, set_dest = solara.use_state("cool")
    view, set_view = solara.use_state("Heat exposure")
    r = solara.use_memo(lambda: heat.cached(trees, refs, list(scope), dest=dest), [trees, refs, scope, dest])
    m = solara.use_memo(lambda: ML.new_map(zoom=15, height=MAP_H), [])
    ref = solara.use_ref({"edges": None, "ref": None, "lst": None, "fit": False, "scope": None})

    def draw():
        ed = r["layers"]["edges"]
        col = "exposure_after" if view == "Heat exposure" else "walkers_after"

        def style(f):
            p = f["properties"]
            if p.get("treated"):
                return {"color": "#1b7f3b", "weight": 6, "opacity": 0.95}
            if view == "Heat exposure":
                return {"color": ML.colour("YlOrRd", p[col], 0.2, 0.9), "weight": 1.6, "opacity": 0.85}
            v = p[col]
            return {"color": ML.colour("Blues", v, 0, 30), "weight": 1 + min(5, v / 6), "opacity": 0.85 if v else 0.25}
        hgj = L.GeoJSON(data=ed, style_callback=style, hover_style={"color": "#000", "weight": 5})

        def heat_info(feature=None, **kw):
            if not feature:
                return
            p = feature["properties"]
            rows = {"Heat exposure, before → after (0–1)": f"{p['exposure_before']:.2f} → {p['exposure_after']:.2f}",
                    "Surface temperature, before → after": f"{p['lst_before']:.1f} → {p['lst_after']:.1f} °C",
                    "Simulated walkers, before → after": f"{p['walkers_before']:.0f} → {p['walkers_after']:.0f}",
                    "New street trees here": "yes" if p.get("treated") else "no"}
            ML.show_feature(m, "Street segment", rows, {
                "what": "Heat exposure combines how sunny the street is at 14:00 and how hot its surface gets (summer 2026 satellite).",
                "read": "0 = shaded and cool, 1 = fully sunny and among the hottest surfaces. Walkers = resident agents passing here."})
        hgj.on_click(heat_info)
        ref.current["edges"] = replace_layer(m, ref.current["edges"], hgj)
        pts = L.GeoJSON(data=r["layers"]["refuges"], point_style={"radius": 9, "color": "#1f2a37", "fillColor": "#2e90fa", "fillOpacity": 1, "weight": 2})
        ML.bind_info(m, pts, "Proposed cooling refuge", {"what": "A civic building (e.g. community centre, library) opened as a cool "
                                                         "place on hot days, chosen to shorten residents' walks the most."})
        ref.current["ref"] = replace_layer(m, ref.current["ref"], pts)
        if ref.current["scope"] != scope:
            ref.current["fit"], ref.current["scope"] = False, scope
            ML.set_focus(m, districts, fit=False)
        if not ref.current["fit"] and ed["features"]:
            xs = [c for f in ed["features"] for c in f["geometry"]["coordinates"]]
            lon, lat = np.mean([c[0] for c in xs]), np.mean([c[1] for c in xs])
            m.center = [float(lat), float(lon)]
            ref.current["fit"] = True
    solara.use_effect(draw, [trees, refs, view, scope, dest])
    S, P = r["summary"], r["proposal"]

    def k(name, f="after"):
        return (S.get(name) or {}).get(f)
    nm = {x["neighbourhood_id"]: x["name"] for x in r.get("by_neighbourhood", [])}
    names = ", ".join(nm.get(i, i) for i in scope) or "–"
    with solara.Row(style={"gap": "0", "align-items": "stretch"}):
        with solara.Div(classes=["ui-side"]):
            solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Heat: proposal → simulation")
            where = "their nearest bus stop" if dest == "bus_stop" else "the nearest park or cooling refuge"
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"Residents of the three top heat-priority neighbourhoods (<b>{names}</b>) "
                        f"walk to {where} at 14:00. Routes avoid sun and hot surfaces; 23% are 65+ and walk slower. "
                        "Street trees are sited on the streets agents actually use and lower both sun and surface temperature.")
            DEST = {"cool": "Nearest cool place", "bus_stop": "Nearest bus stop"}
            inv_dest = {v: k for k, v in DEST.items()}
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Residents walk to")
            solara.ToggleButtonsSingle(value=DEST[dest], on_value=lambda v: set_dest(inv_dest[v]), values=list(DEST.values()), dense=True)
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Street trees (metres of street)")
            solara.ToggleButtonsSingle(value=trees, on_value=set_trees, values=[0, 500, 1000, 2000, 3000], dense=True)
            if dest == "cool":
                solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Cooling refuges opened")
                solara.ToggleButtonsSingle(value=refs, on_value=set_refs, values=[0, 1, 2, 4, 6], dense=True)
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"≈{P['trees']:.0f} trees, {P['kerb_m']:.0f} m kerb used; "
                        f"surface temp. on planted streets {P['lst_change_on_treated_c']:+.2f} °C")
            people = sum(x.get("people", 0) for x in r.get("by_neighbourhood", []))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Effect on residents: today → with this proposal")
            def ba(name, nd=0, unit=""):
                return f"{fmt(k(name, 'before'), nd, unit)} → {fmt(k(name), nd, unit)}"
            if dest == "bus_stop":
                solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"Simulated walks of {people:,.0f} residents at 14:00 on a summer day to "
                            "their nearest bus stop, on today's streets and with the new street trees (same people, 6 runs). Trees are placed "
                            "on the streets people actually use to reach the stops.")
                with solara.Div(classes=["kpis"]):
                    Kpi("share of the walk in shade", ba("all_shade_pct", 0, "%"), fmt(k("all_shade_pct", "delta"), 0, " points", True), "up", "")
                    Kpi("minutes in the sun per walk", ba("all_sun_min", 1), fmt(k("all_sun_min", "delta"), 1, " min", True), "down", "")
                    Kpi("minutes in the sun, over-65s", ba("vuln_sun_min", 1), fmt(k("vuln_sun_min", "delta"), 1, " min", True), "down", "")
                    Kpi("heat dose per walk (min)", ba("all_heat_min", 1), fmt(k("all_heat_min", "delta"), 1, " min", True), "down", "")
            if dest != "bus_stop":
              solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"Simulated walks of {people:,.0f} residents at 14:00 on a summer day to "
                        "the nearest park or cool place, on today's streets and with the proposal (same people, 6 runs). A walk counts as a "
                        "<b>high heat dose</b> when it adds up to 8+ minutes of full exposure (sun × surface heat).")
            with solara.Div(classes=["kpis"]) if dest != "bus_stop" else solara.Div(style={"display": "none"}):
                Kpi("over-65s whose walk is a high heat dose", ba("vuln_over_threshold_pct", 0, "%"),
                    fmt(k("vuln_over_threshold_pct", "delta"), 0, " points", True), "down", "")
                Kpi("everyone whose walk is a high heat dose", ba("all_over_threshold_pct", 0, "%"),
                    fmt(k("all_over_threshold_pct", "delta"), 0, " points", True), "down", "")
                Kpi("heat dose per walk, over-65s (min)", ba("vuln_heat_min", 1), fmt(k("vuln_heat_min", "delta"), 1, " min", True), "down", "")
                Kpi("residents with a cool place ≤5 min away", ba("all_within5_pct", 0, "%"),
                    fmt(k("all_within5_pct", "delta"), 0, " points", True), "up", "")
            NeighbourhoodMeasures(scope)
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Map")
            solara.ToggleButtonsSingle(value=view, on_value=set_view, values=["Heat exposure", "Where agents walk"], dense=True)
            solara.HTML(tag="div", unsafe_innerHTML=legend_html(
                ([("cool street", ML.colour("YlOrRd", 0.2, 0.2, 0.9)), ("hot street", ML.colour("YlOrRd", 0.9, 0.2, 0.9))]
                 if view == "Heat exposure" else [("few walkers", ML.colour("Blues", 3, 0, 30)), ("many walkers", ML.colour("Blues", 30, 0, 30))])
                + [("new street trees", "#1b7f3b"), ("new cooling refuge", "#2e90fa")]))
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Heat dose = minutes walked × exposure (½ share of the street in sun "
                        "at 14:00, ½ satellite surface temperature). The 8-minute limit is an assumption, not a clinical threshold.")
        with solara.Div(classes=["ui-main"]):
            solara.display(m)
    HeatCharts(sweep if scope == default_scope and dest == "cool" else None, r)


@solara.component
def HeatCharts(sweep, r):
    if sweep is None:
        nb = pd.DataFrame(r["by_neighbourhood"])
        if len(nb):
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Budget sweep chart is precomputed for the city-wide top 3; "
                        "the table shows the selected districts' top 3.")
            solara.DataFrame(pd.DataFrame({"neighbourhood": nb.name, "people": nb.people.round(0),
                                           "dose Δ (min)": nb.all_heat_min_delta.round(2),
                                           "65+ over threshold Δ (pp)": nb.vuln_over_threshold_pct_delta.round(1),
                                           "≤5 min Δ (pp)": nb.all_within5_pct_delta.round(1)}), items_per_page=6)
        return
    sw = pd.DataFrame(sweep["sweep"])
    fig = go.Figure()
    t = sw[(sw.refuges == 0)].sort_values("tree_budget_m")
    fig.add_scatter(x=t.tree_budget_m, y=-t.vuln_over_threshold_pct_delta, mode="lines+markers", name="street trees (x = metres)",
                    line_color="#1b7f3b")
    rf = sw[(sw.tree_budget_m == 0)].sort_values("refuges")
    fig.add_scatter(x=rf.refuges * 500, y=-rf.vuln_over_threshold_pct_delta, mode="lines+markers", name="refuges (x = count × 500)",
                    line_color="#2e90fa")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), template="simple_white", legend=dict(orientation="h", y=-0.2),
                      title="65+ trips moved below the dose threshold (percentage points)", xaxis_title="budget")
    nb = pd.DataFrame(r["by_neighbourhood"])
    with solara.Columns([3, 2]):
        solara.FigurePlotly(fig)
        if len(nb):
            tbl = pd.DataFrame({"neighbourhood": nb.name, "people": nb.people.round(0),
                                "dose Δ (min)": nb.all_heat_min_delta.round(2), "65+ over threshold Δ (pp)": nb.vuln_over_threshold_pct_delta.round(1),
                                "≤5 min Δ (pp)": nb.all_within5_pct_delta.round(1)})
            solara.DataFrame(tbl, items_per_page=6)


# ------------------------------------------------------------------------------------------------------------ Flood
@solara.component
def Flood(districts=()):
    flood.register_layers()
    ex_city = solara.use_memo(flood.city_exposure, [])
    ids = set(overview.nb_ids(districts))

    def sel_exposure():
        if not districts:
            return ex_city
        e = flood.exposure_by_neighbourhood()
        e = e[e.neighbourhood_id.isin(ids)]
        pop = overview.headline(districts).get("population") or 0
        out = {}
        for z in ("flood_50", "flood_100", "flood_1000"):
            ppl = float(e[e.zone == z].people.sum())
            out[z] = {"people": ppl, "pop_pct": 100 * ppl / pop if pop else None}
        return out
    ex = solara.use_memo(sel_exposure, [districts])
    base, set_base = solara.use_state("runoff")
    key, set_key = solara.use_state(flood.LOCAL_KEY)
    show_form, set_show_form = solara.use_state(False)
    d_start, set_d_start = solara.use_state(datetime.date(2025, 10, 1))
    d_end, set_d_end = solara.use_state(datetime.date(2025, 12, 31))
    cloud, set_cloud = solara.use_state(20)
    nimg, set_nimg = solara.use_state(6)
    msg, set_msg = solara.use_state("")
    pers = solara.use_memo(flood.periods, [key, msg])
    raster = flood.layer_id(base, key) if base != "none" else "none"

    def work():
        def prog(frac, text):
            task.progress = int(frac * 100)
            set_msg(text)
        r = flood.spectral_range(d_start, d_end, cloud, nimg, progress=prog)
        return r["key"]
    task = solara.lab.use_task(work, dependencies=None, raise_error=False)

    def after_run():
        if task.finished and task.value:
            flood.register_layers()
            set_key(task.value)
            set_show_form(False)
    solara.use_effect(after_run, [task.finished, task.value])
    m = solara.use_memo(lambda: ML.new_map(zoom=13, height=MAP_H, fit_municipality=True), [])
    solara.use_effect(lambda: ML.set_focus(m, districts), [districts])
    ref = solara.use_ref({"r": None, "fixed": False})

    def draw():
        if not ref.current["fixed"]:
            for lid in ("rivers", "flood_1000", "flood_100", "flood_50"):
                m.add_layer(ML.build_layer(lid, m))
            ref.current["fixed"] = True
        new = ML.build_layer(raster, m) if raster != "none" and raster in LY.BY_ID else None
        old = ref.current["r"]
        rest = [l for l in m.layers if l is not old]
        tiles = [l for l in rest if isinstance(l, L.TileLayer)]
        others = [l for l in rest if not isinstance(l, L.TileLayer)]
        m.layers = tuple(tiles + ([new] if new is not None else []) + others)     # raster under the zones
        ref.current["r"] = new
    solara.use_effect(draw, [raster])
    try:
        net = flood.network_impact()
        if districts:
            net = net[net.neighbourhood_id.isin(ids)]
        cut = net.groupby("dest").cut_off_people.sum().to_dict()
        km = float(net.flooded_street_km.iloc[0]) if len(net) else None
    except Exception:
        net, cut, km = pd.DataFrame(), {}, None
    with solara.Row(style={"gap": "0", "align-items": "stretch"}):
        with solara.Div(classes=["ui-side"]):
            solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Flood exposure" + (
                "" if not districts else " · " + ", ".join(overview.ORD[d] for d in districts) + (" district" if len(districts) == 1 else " districts")))
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Official Flood Directive hazard zones combined with modelled residents, "
                        "the street network, and wet-season Sentinel-2 indices.")
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Residents in hazard zone")
            with solara.Div(classes=["kpis"]):
                for lid, lab in (("flood_50", "50-year"), ("flood_100", "100-year"), ("flood_1000", "1000-year")):
                    e = ex.get(lid, {})
                    Kpi(f"{lab} zone ({fmt(e.get('pop_pct'), 1, '%')})", fmt(e.get("people"), 0))
                Kpi("streets in 100-yr zone (km)", fmt(km, 1))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="If 100-yr zone streets close")
            with solara.Div(classes=["kpis"]):
                Kpi("residents cut off from a bus stop (≤1.5 km)", fmt(cut.get("bus_stop"), 0))
                Kpi("residents cut off from a park", fmt(cut.get("park"), 0))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Satellite layer")
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Satellites record how much light each 10–20 m square reflects in several "
                        "colours, including infrared we cannot see. Water, plants and concrete reflect these colours differently, so simple ratios "
                        "between two colours (<i>spectral indices</i>) highlight one of them. By default the images are the local <b>wet season</b> "
                        "set (Jan–Apr 2026), when soils are wettest; you can compute any other period below. Click the map to read a value.")
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Image period")
            labels = {p["key"]: p["label"] for p in pers}
            lab2key = {v: k for k, v in labels.items()}
            if key not in labels:
                labels[key] = key
            solara.Select(label="", value=labels[key], values=list(labels.values()), on_value=lambda v: set_key(lab2key[v]),
                          dense=True)
            if not show_form and not task.pending:
                solara.Button("New period…", on_click=lambda: set_show_form(True), text=True, small=True, color="primary")
            if show_form or task.pending:
                with solara.Div(style={"border": "1px solid #e6e8eb", "border-radius": "8px", "padding": "8px", "margin": "4px 0 8px"}):
                    solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Downloads Sentinel-2 L2A images for the period from the Planetary "
                                "Computer STAC (the clearest ones, up to the limit), then computes NDWI, MNDWI, NDVI, NDBI and runoff "
                                "susceptibility. Takes about 1–3 minutes; images already downloaded are reused.")
                    solara.lab.InputDate(d_start, on_value=lambda v: v and set_d_start(v), label="From", min_date="2017-01-01")
                    solara.lab.InputDate(d_end, on_value=lambda v: v and set_d_end(v), label="To", min_date="2017-01-01")
                    solara.SliderInt(f"Max. cloud cover: {cloud}%", value=cloud, on_value=set_cloud, min=5, max=60, step=5)
                    solara.SliderInt(f"Max. images: {nimg}", value=nimg, on_value=set_nimg, min=1, max=10)
                    if task.pending:
                        solara.ProgressLinear(task.progress if task.progress else True)
                        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=msg or "starting…")
                    else:
                        with solara.Row():
                            solara.Button("Download & compute", on_click=task, color="primary", small=True)
                            solara.Button("Cancel", on_click=lambda: set_show_form(False), text=True, small=True)
                    if task.error:
                        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"<span style='color:#b42318'>Failed: {task.exception}</span>")
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Index")
            nice = {"none": "None", "runoff": "Runoff", "s2_mndwi": "MNDWI", "s2_ndwi": "NDWI", "s2_ndbi": "NDBI", "s2_ndvi": "NDVI"}
            opts = [k for k in nice if k == "none" or flood.layer_id(k, key) in LY.BY_ID]
            inv = {nice[k]: k for k in opts}
            solara.ToggleButtonsSingle(value=nice[base], on_value=lambda v: set_base(inv[v]), values=[nice[k] for k in opts], dense=True,
                                       style={"flex-wrap": "wrap"})
            if raster != "none" and raster in LY.BY_ID:
                solara.HTML(tag="div", unsafe_innerHTML=ML.index_card_html(raster, open_more=False))
            solara.HTML(tag="div", unsafe_innerHTML=legend_html([("50-yr zone", "#1e88e5"), ("100-yr zone", "#42a5f5"), ("1000-yr zone", "#90caf9"),
                                                                ("streams", "#0277bd")]))
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Runoff susceptibility = 0.4 built-up (NDBI) + 0.2 low vegetation + "
                        "0.2 soil group + 0.2 stream proximity. A screening proxy, not a hydraulic model.")
        with solara.Div(classes=["ui-main"]):
            solara.display(m)
    nb = flood.exposure_by_neighbourhood()
    if districts:
        nb = nb[nb.neighbourhood_id.isin(ids)]
    if len(nb):
        p = nb.pivot_table(index="neighbourhood_id", columns="zone", values="people", aggfunc="sum").fillna(0)
        t = pd.DataFrame({"neighbourhood": [nb_names().get(i, i) for i in p.index]}, index=p.index)
        for z, lab in (("flood_100", "people, 100-yr"), ("flood_1000", "people, 1000-yr")):
            t[lab] = p.get(z, 0).round(0)
        if len(net):
            t["cut off from bus stop"] = net[net.dest == "bus_stop"].set_index("neighbourhood_id").cut_off_people.reindex(p.index).fillna(0).round(0)
        ro = flood.runoff_by_neighbourhood(key)
        if len(ro):
            t["runoff score (selected period)"] = ro.set_index("neighbourhood_id").runoff_mean.reindex(p.index).round(2)
        solara.DataFrame(t.sort_values("people, 1000-yr", ascending=False).reset_index(drop=True), items_per_page=8)


# ------------------------------------------------------------------------------------------------------------ Timelapse
def _data_url(path):
    import base64
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


@solara.component
def Timelapse(districts=()):
    sensor_lab = {k: v["label"] for k, v in timelapse.SENSORS.items()}
    lab2sensor = {v: k for k, v in sensor_lab.items()}
    sensor, set_sensor = solara.use_state("sentinel-2")
    product, set_product = solara.use_state("ndvi")
    first = timelapse.SENSORS[sensor]["first_year"]
    years, set_years = solara.use_state((2018, datetime.date.today().year))
    season, set_season = solara.use_state("Summer (Jun–Sep)")
    step, set_step = solara.use_state("yearly")
    cloud, set_cloud = solara.use_state(20)
    nimg, set_nimg = solara.use_state(3)
    speed, set_speed = solara.use_state(900)
    msg, set_msg = solara.use_state("")
    current, set_current = solara.use_state(None)          # key of the timelapse on screen
    fidx, set_fidx = solara.use_state(0)
    prods = {k: v[0] for k, v in timelapse.PRODUCTS.items() if sensor in v[4]}
    if product not in prods:
        product = "ndvi"
    lab2prod = {v: k for k, v in prods.items()}
    y0, y1 = max(years[0], first), years[1]
    try:
        n_steps = len(timelapse.windows(y0, y1, timelapse.SEASONS[season], step))
        step_err = ""
    except ValueError as e:
        n_steps, step_err = 0, str(e)

    def work():
        def prog(frac, text):
            task.progress = int(frac * 100)
            set_msg(text)
        return timelapse.build(sensor, product, y0, y1, season, step, districts, cloud, nimg, speed, progress=prog)["key"]
    task = solara.lab.use_task(work, dependencies=None, raise_error=False)

    def after():
        if task.finished and task.value:
            set_current(task.value)
            set_fidx(0)
    solara.use_effect(after, [task.finished, task.value])
    meta = timelapse.get(current) if current else None
    done = [t for t in timelapse.listing() if t.get("status") == "done"]
    m = solara.use_memo(lambda: ML.new_map(height="calc(100vh - 160px)", fit_municipality=True), [])
    solara.use_effect(lambda: ML.set_focus(m, districts), [districts])
    ref = solara.use_ref({"ov": None})

    def draw():
        old = ref.current["ov"]
        if old is not None and old in m.layers:
            m.remove_layer(old)
        ref.current["ov"] = None
        if not meta or not meta.get("frames"):
            return
        f = meta["frames"][min(fidx, len(meta["frames"]) - 1)]
        ov = L.ImageOverlay(url=_data_url(timelapse.path(meta["key"], f["overlay"])), bounds=meta["bounds"], opacity=0.85)
        m.add_layer(ov)
        ref.current["ov"] = ov
        ML.show_info(m, ML._info_html(f"{meta['product_label']} · {f['label']}", {
            "what": f"Median of {f['n_images']} {meta['sensor_label']} image(s): {', '.join(f['dates'])}.",
            "read": ML.explain.about("s2_" + meta["params"]["product"]).get("read", "") if meta["params"]["product"] in ("ndwi", "mndwi", "ndbi")
            else (ML.explain.about("ndvi").get("read", "") if meta["params"]["product"] == "ndvi" else "")},
            value=(f"area mean {f['mean']:.3f} {meta['unit']}".strip() if f.get("mean") is not None else None)))
    solara.use_effect(draw, [current, fidx])

    with solara.Row(style={"gap": "0", "align-items": "stretch"}):
        with solara.Div(classes=["ui-side"]):
            solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Satellite timelapse")
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Downloads new satellite images for every time step, removes clouds, "
                        "combines them and computes the chosen index, then plays the steps as an animation. Area: <b>"
                        + timelapse.area(districts)[1] + "</b> (change it with the district selector at the top).")
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Satellite")
            solara.ToggleButtonsSingle(value=sensor_lab[sensor], on_value=lambda v: set_sensor(lab2sensor[v]), values=list(sensor_lab.values()),
                                       dense=True)
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Sentinel-2: from 2017, 10–20 m detail. Landsat: from 1984, 30 m, "
                        "and it also measures surface temperature.")
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Show")
            solara.ToggleButtonsSingle(value=prods[product], on_value=lambda v: set_product(lab2prod[v]), values=list(prods.values()), dense=True)
            if product != "true_color":
                solara.HTML(tag="div", unsafe_innerHTML=ML.index_card_html({"ndvi": "ndvi", "lst": "lst"}.get(product, "s2_" + product)))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Period")
            solara.SliderRangeInt(f"Years: {y0}–{y1}", value=(max(years[0], first), years[1]), on_value=set_years, min=first,
                                  max=datetime.date.today().year)
            solara.Select("Season", value=season, values=list(timelapse.SEASONS), on_value=set_season, dense=True)
            solara.ToggleButtonsSingle(value=step, on_value=set_step, values=["yearly", "monthly"], dense=True)
            solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=(f"<span style='color:#b42318'>{step_err}</span>" if step_err else
                        f"{n_steps} time steps: one composite per {'year (same season each year, so changes are not just seasons)' if step == 'yearly' else 'month'}."))
            solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Images")
            solara.SliderInt(f"Max. cloud cover: {cloud}%", value=cloud, on_value=set_cloud, min=5, max=60, step=5)
            solara.SliderInt(f"Images per step: {nimg} (clearest first)", value=nimg, on_value=set_nimg, min=1, max=6)
            solara.SliderInt(f"Speed: {speed} ms per frame", value=speed, on_value=set_speed, min=300, max=2000, step=100)
            if task.pending:
                solara.ProgressLinear(task.progress if task.progress else True)
                solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=msg or "starting…")
            else:
                solara.Button("Download & build timelapse", on_click=task, color="primary", disabled=bool(step_err) or n_steps == 0)
                solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="About 3–6 s per image; images already downloaded are reused.")
            if task.error:
                solara.HTML(tag="div", classes=["note"], unsafe_innerHTML=f"<span style='color:#b42318'>Failed: {task.exception}</span>")
            if done:
                solara.HTML(tag="div", classes=["grp"], unsafe_innerHTML="Saved timelapses")
                labs = {t["key"]: f"{t['area']} · {t['product_label']} · {t['frames'][0]['label']} → {t['frames'][-1]['label']}" for t in done}
                inv = {v: k for k, v in labs.items()}
                solara.Select("Open", value=labs.get(current), values=list(labs.values()),
                              on_value=lambda v: (set_current(inv[v]), set_fidx(0)), dense=True)
        with solara.Column(classes=["ui-main"], style={"gap": "4px", "justify-content": "flex-start", "align-self": "flex-start"}):
            solara.display(m)
            if meta and meta.get("frames"):
                fr = meta["frames"]
                solara.SliderInt(f"Frame: {fr[min(fidx, len(fr) - 1)]['label']}", value=min(fidx, len(fr) - 1), on_value=set_fidx,
                                 min=0, max=len(fr) - 1, tick_labels=[f["label"].split()[-1] for f in fr] if len(fr) <= 16 else False)
    if meta and meta.get("status") == "done":
        with solara.Columns([3, 2]):
            gif = timelapse.path(meta["key"], "timelapse.gif")
            with solara.Column():
                solara.Image(gif.read_bytes(), width="100%")
                solara.FileDownload(gif.read_bytes(), filename=f"athens_{meta['params']['product']}_timelapse.gif", label="Download GIF")
            with solara.Column():
                fr = [f for f in meta["frames"] if f.get("mean") is not None]
                if fr:
                    fig = go.Figure(go.Scatter(x=[f["label"] for f in fr], y=[f["mean"] for f in fr], mode="lines+markers",
                                               line_color="#1f2a37"))
                    fig.update_layout(height=280, margin=dict(l=10, r=10, t=36, b=10), template="simple_white",
                                      title=f"{meta['product_label']}: mean over {meta['area']}", yaxis_title=meta["unit"] or None)
                    solara.FigurePlotly(fig)
                if meta.get("skipped"):
                    solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Skipped: " + "; ".join(
                        f"{x['label']} ({x['reason']})" for x in meta["skipped"]))


# ------------------------------------------------------------------------------------------------------------ Coming next
ROADMAP = [
    {"icon": "🗑️", "title": "Waste collection",
     "question": "Where do bins overflow, and how would a new collection route or schedule change service across neighbourhoods?",
     "show": "Bin locations and fill levels, collection routes and times, complaints per street.",
     "abm": "Collection trucks as agents on the street network: test new routes, times and depot locations, and see who gets "
            "faster or slower service, plus the effect on traffic.",
     "data": ["City of Athens cleaning department: bins, routes, schedules (to request)", "Citizen complaints (e.g. 1595 / city apps, to request)",
              "OSM street network and Attica traffic detectors (in the platform)"]},
    {"icon": "🛡️", "title": "Safety and reported incidents",
     "question": "Where do people feel unsafe, and do lighting, footfall or new uses change that?",
     "show": "Reported incidents aggregated per neighbourhood or street segment (never individuals), street lighting, footfall.",
     "abm": "Pedestrian agents choose routes partly by perceived safety, showing how lighting or active ground floors change "
            "where people walk at night.",
     "data": ["Hellenic Police crime statistics, aggregated (to confirm granularity)", "Street-light inventory (to request)",
              "Walking network and population model (in the platform)"],
     "note": "Aggregated data only, with care to avoid stigmatising neighbourhoods."},
    {"icon": "🏘️", "title": "Gentrification and displacement",
     "question": "Which neighbourhoods are changing fastest, and who is at risk of being priced out?",
     "show": "Change in residents, age and household mix between censuses, short-term rentals, rent and sale prices.",
     "abm": "Household agents decide whether to stay or move as rents and amenities change. Test how a new park or a reopened "
            "street could raise local prices.",
     "data": ["ELSTAT census 2011 vs 2021", "Inside Airbnb, Athens (short-term rental listings)", "Rent / price indices (to source)"]},
    {"icon": "🤝", "title": "Social impact",
     "question": "Who gains and who loses from each proposal, by age, income and mobility?",
     "show": "Every case-study result split by population group: over-65s, children, low-income areas, people without a car.",
     "abm": "Already partly built: the Heat case reports over-65s separately. Extend every simulation with equity "
            "indicators and a fairness score per proposal.",
     "data": ["ELSTAT census age and household structure (in part in the platform)", "Income or deprivation proxy per neighbourhood (to source)",
              "Results of the existing case studies"]},
    {"icon": "🚨", "title": "Emergency awareness: earthquakes and attacks",
     "question": "How quickly can residents reach safe open space, and which streets become bottlenecks?",
     "show": "Seismic zones and past earthquakes (in the platform), designated safe assembly areas, evacuation routes, "
            "vulnerable buildings.",
     "abm": "Evacuation simulation: resident agents walk to the nearest safe area while streets may be blocked. Find "
            "neighbourhoods that are too far from safety and bottleneck streets, and test opening new assembly areas.",
     "data": ["Seismic hazard zones and earthquakes 2006–2026 (in the platform)",
              "Civil protection safe assembly areas (City of Athens / OASP, to confirm)",
              "Building age and height (Copernicus, in part in the platform)", "Walking network and population model (in the platform)"]},
]


@solara.component
def ComingNext(districts=()):
    cards = []
    for r in ROADMAP:
        data = "".join(f"<li>{html.escape(d)}</li>" for d in r["data"])
        note = f"<div class='note' style='margin-top:6px'><i>{html.escape(r['note'])}</i></div>" if r.get("note") else ""
        cards.append(
            "<div class='rm-card'>"
            f"<div class='rm-head'><span class='rm-icon'>{r['icon']}</span><span class='rm-title'>{html.escape(r['title'])}</span>"
            "<span class='rm-status'>Planned</span></div>"
            f"<div class='rm-q'>{html.escape(r['question'])}</div>"
            f"<div class='rm-row'><b>On the map:</b> {html.escape(r['show'])}</div>"
            f"<div class='rm-row'><b>What the simulation adds:</b> {html.escape(r['abm'])}</div>"
            f"<div class='rm-row'><b>Data needed:</b><ul>{data}</ul></div>{note}</div>")
    solara.Style(""".rm-wrap {padding: 8px 12px 24px; max-width: 1200px;}
.rm-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 14px;}
.rm-card {border: 1px solid #e6e8eb; border-radius: 10px; padding: 14px 16px; background: #fff; font-size: 13px; line-height: 1.45; color: #1f2a37;}
.rm-head {display: flex; align-items: center; gap: 8px; margin-bottom: 6px;}
.rm-icon {font-size: 20px;} .rm-title {font-weight: 700; font-size: 15px; flex: 1;}
.rm-status {background: #eef2f6; color: #344054; border-radius: 10px; padding: 1px 9px; font-size: 11px; font-weight: 600;}
.rm-q {color: #344054; font-style: italic; margin-bottom: 6px;}
.rm-row {margin: 5px 0;} .rm-row ul {margin: 2px 0 0 18px; padding: 0;} .rm-row li {margin: 1px 0;}""")
    with solara.Div(classes=["rm-wrap"]):
        solara.HTML(tag="h3", classes=["title"], unsafe_innerHTML="Coming next")
        solara.HTML(tag="div", classes=["note"], unsafe_innerHTML="Planned modules. Each reuses the platform's map, population model and "
                    "agent-based simulation; the data marked <i>to request</i> or <i>to confirm</i> is what is still missing.")
        solara.HTML(tag="div", classes=["rm-grid"], unsafe_innerHTML="".join(cards))


# ------------------------------------------------------------------------------------------------------------ app shell
PAGES = {"Overview": Overview, "Vas. Olgas": Kallimarmaro, "Heat": Heat, "Flood": Flood, "Timelapse": Timelapse,
         "Coming next": ComingNext}
HEADER_CSS = """
.app-head {display:flex; align-items:center; gap:24px; background:#1f2a37; color:#fff; padding:0 16px; height:52px; margin:-8px -8px 8px;}
.app-head .brand {font-weight:700; font-size:16px; white-space:nowrap;}
.app-head .brand small {font-weight:400; opacity:.7; margin-left:6px;}
.app-head .v-btn-toggle {background: transparent !important;}
.app-head .v-btn {color:#cfd6df !important; background: transparent !important; text-transform:none; letter-spacing:0; font-size:14px;}
.app-head .v-btn.v-btn--active {color:#fff !important; border-bottom: 2px solid #fdb022 !important;}
.solara-content-main {padding: 8px !important;}
.app-head {overflow:hidden; min-width:0;}
.app-head .brand small {display:none;}
.app-head .v-btn-group, .app-head .v-btn-toggle {flex:1 1 auto; min-width:0; overflow-x:auto; flex-wrap:nowrap !important;}
.app-head .v-btn {padding:0 8px !important; min-width:0 !important;}
.app-head .dsel {flex:0 0 230px; margin-left:8px; background:#fff; border-radius:6px; padding:0 6px; color:#1f2a37;}
html, body, .v-application {overflow-x:hidden;}
@media (min-width: 1300px) {.app-head .brand small {display:inline;} .app-head .dsel {flex-basis:300px;}}
.app-head .dsel .v-input {margin:0; padding:0;}
.app-head .dsel .v-field__input {min-height:34px !important; padding-top:2px !important; padding-bottom:2px !important; font-size:13px;}
.app-head .dsel .v-input__details {display:none;}
"""


@solara.component
def Page():
    solara.Style(CSS + HEADER_CSS)
    solara.Title("Athens Urban Intelligence")
    tab, set_tab = solara.use_state("Overview")
    sel, set_sel = solara.use_state([])                        # district labels; empty = whole municipality
    labels = overview.districts()
    lab2dk = {v: k for k, v in labels.items()}
    dks = overview.norm_districts([lab2dk[x] for x in sel if x in lab2dk])
    with solara.Column(style={"padding": "0", "gap": "0"}):
        with solara.Div(classes=["app-head"]):
            solara.HTML(tag="div", classes=["brand"], unsafe_innerHTML="Athens Urban Intelligence<small>agent-based city what-ifs</small>")
            solara.ToggleButtonsSingle(value=tab, on_value=set_tab, values=list(PAGES), dense=True)
            with solara.Div(classes=["dsel"]):
                solara.SelectMultiple("Districts (all of Athens)" if not sel else "Districts", values=sel, all_values=list(labels.values()),
                                      on_value=set_sel, dense=True)
        PAGES[tab](districts=dks)


@solara.component
def Layout(children=[]):
    # no app bar / sidebar chrome: the Page draws its own header
    with solara.Column(style={"padding": "0"}, children=children):
        pass
