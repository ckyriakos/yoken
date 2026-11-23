"""Maps for the demo: basemap switcher, municipality focus, an info box (hover/click on features, click on rasters), and catalogue
layers built at a chosen level of detail. Layer data is cached per process, so a layer is converted once and reused by every page."""
from __future__ import annotations

import base64
import html
import threading
import time
from functools import lru_cache

import ipyleaflet as L
import ipywidgets as W
import leafmap
import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import to_hex
from pyproj import Transformer

from .. import explain
from .. import layers as LY
from ..services import overview

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"
BASEMAPS = {
    "Light grey": [dict(url=f"{ESRI}/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}", max_native_zoom=16,
                        attribution="Esri, HERE, Garmin, © OpenStreetMap contributors"),
                   dict(url=f"{ESRI}/Canvas/World_Light_Gray_Reference/MapServer/tile/{{z}}/{{y}}/{{x}}", max_native_zoom=16, opacity=0.8)],
    "Streets": [dict(url="https://tile.openstreetmap.org/{z}/{x}/{y}.png", max_native_zoom=19, attribution="© OpenStreetMap contributors")],
    "Satellite": [dict(url=f"{ESRI}/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}", max_native_zoom=19,
                       attribution="Esri, Maxar, Earthstar Geographics"),
                  dict(url=f"{ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}", max_native_zoom=19, opacity=0.9)],
    "Topographic": [dict(url=f"{ESRI}/World_Topo_Map/MapServer/tile/{{z}}/{{y}}/{{x}}", max_native_zoom=19,
                         attribution="Esri, HERE, Garmin, © OpenStreetMap contributors")],
}
DETAIL_TOL = {"Fast": 1e-4, "Standard": 3e-5, "Full": None}      # simplification tolerance in degrees (~10 m, ~3 m, none)
DETAIL_DIGITS = {"Fast": 4, "Standard": 5, "Full": 6}            # coordinate decimals sent to the browser (~10 m, ~1 m, ~0.1 m)
HOVER_MAX_FEATURES = 1500     # bigger layers answer clicks only: hover messages while panning make the map sluggish
TO3857 = Transformer.from_crs(4326, 3857, always_xy=True)
INFO_HINT = "<div style='color:#5b6573'>Click a feature (or hover small ones), or click anywhere on a raster layer, for details.</div>"


# ------------------------------------------------------------------------------------------------------------ helpers
def colour(cmap, x, vmin=0.0, vmax=1.0):
    t = 0.0 if vmax == vmin else min(1.0, max(0.0, (x - vmin) / (vmax - vmin)))
    return to_hex(colormaps[cmap](t))


@lru_cache(maxsize=64)
def _png_url(layer_id):
    """Overlay as a palette PNG (256 colours + alpha): several times smaller than RGBA, so pages load and pan faster."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(overview.raster_png(layer_id))).convert("RGBA")
    buf = io.BytesIO()
    im.quantize(colors=256, method=Image.Quantize.FASTOCTREE).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def round_coords(obj, nd):
    """Round all coordinates of a GeoJSON dict in place (smaller payload, faster drawing)."""
    def r(c):
        return [round(c[0], nd), round(c[1], nd)] if isinstance(c[0], (int, float)) else [r(x) for x in c]
    feats = obj.get("features", [obj]) if isinstance(obj, dict) else []
    for f in feats:
        g = f.get("geometry") or {}
        if g.get("coordinates") is not None:
            g["coordinates"] = r(g["coordinates"])
    return obj


@lru_cache(maxsize=128)
def _geojson(layer_id, tol, nd=5):
    return round_coords(overview.geojson(layer_id, simplify=tol), nd)


@lru_cache(maxsize=32)
def _heat(layer_id):
    return overview.heat_points(layer_id)


@lru_cache(maxsize=1)
def municipality_geojson():
    """(mask with a hole for the municipality, outline, bounds) in WGS84."""
    import geopandas as gpd
    from shapely.geometry import box, mapping
    g = gpd.GeoSeries([overview.municipality().buffer(-overview.CLIP_BUFFER_M + 20)], crs=2100).to_crs(4326).iloc[0]
    world = box(19.0, 34.5, 29.5, 42.0)
    mask = world.difference(g)
    b = g.bounds
    return ({"type": "Feature", "properties": {}, "geometry": mapping(mask)},
            {"type": "Feature", "properties": {}, "geometry": mapping(g.boundary)}, [[b[1], b[0]], [b[3], b[2]]])


# ------------------------------------------------------------------------------------------------------------ info box
def _info_html(title, ex=None, rows=(), value=None):
    e = ex or {}
    parts = [f"<div style='font-weight:700;font-size:13px;color:#1f2a37;margin-bottom:3px'>{html.escape(title)}</div>"]
    if value is not None:
        parts.append(f"<div style='font-size:18px;font-weight:600;color:#1f2a37'>{html.escape(value)}</div>")
    if e.get("what"):
        parts.append(f"<div style='margin:3px 0'>{html.escape(e['what'])}</div>")
    if e.get("proxy"):
        parts.append(f"<div style='margin:3px 0'><b>Stands for:</b> {html.escape(e['proxy'])}</div>")
    if e.get("read"):
        parts.append(f"<div style='margin:3px 0'><b>How to read:</b> {html.escape(e['read'])}</div>")
    if rows:
        parts.append("<table style='margin-top:4px;border-collapse:collapse'>" + "".join(
            f"<tr><td style='color:#5b6573;padding:1px 8px 1px 0;vertical-align:top'>{html.escape(k)}</td>"
            f"<td style='padding:1px 0'>{html.escape(v)}</td></tr>" for k, v in rows) + "</table>")
    if e.get("when"):
        parts.append(f"<div style='margin-top:4px;color:#5b6573'><b>When:</b> {html.escape(e['when'])}</div>")
    if e.get("source"):
        parts.append(f"<div style='color:#5b6573'><b>Source:</b> {html.escape(e['source'])}</div>")
    return ("<div style='font:12px/1.35 system-ui,sans-serif;max-width:300px;max-height:360px;overflow:auto;padding:2px 4px'>"
            + "".join(parts) + "</div>")


def _fmt_val(v):
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"
    return str(v)


def feature_rows(props, labels=None, limit=10):
    labels = {**explain.FIELD_LABELS, **(labels or {})}
    rows = []
    for k, v in props.items():
        if k in explain.SKIP_FIELDS or k.endswith("_html") or k.startswith("spark") or isinstance(v, (list, dict)):
            continue
        if v in (None, "", "None", "nan") or k.startswith("_") or k == "style":
            continue
        if k not in labels and labels is not explain.FIELD_LABELS and len(rows) >= limit:
            continue
        rows.append((labels.get(k, k), _fmt_val(v)))
        if len(rows) >= limit:
            break
    return rows


def show_info(m, html_text):
    if getattr(m, "_info", None) is not None:
        m._info.value = html_text


def sparkline_svg(values, label="", unit="", w=280, h=56, color="#b42318"):
    """Tiny inline SVG line chart for the info box."""
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return ""
    lo, hi = min(v), max(v)
    span = (hi - lo) or 1
    n = len(values)
    pts, seg = [], []
    for i, x in enumerate(values):
        if x is None:
            if seg: pts.append(seg); seg = []
            continue
        seg.append(f"{i / (n - 1) * (w - 4) + 2:.1f},{h - 14 - (x - lo) / span * (h - 22):.1f}")
    if seg: pts.append(seg)
    lines = "".join(f"<polyline fill='none' stroke='{color}' stroke-width='1.6' points='{' '.join(sg)}'/>" for sg in pts)
    return (f"<div style='margin-top:5px;color:#5b6573;font-size:11px'>{html.escape(label)}</div>"
            f"<svg width='{w}' height='{h}' style='display:block'>{lines}"
            f"<text x='2' y='{h - 2}' font-size='10' fill='#5b6573'>min {lo:.1f}{unit}</text>"
            f"<text x='{w - 2}' y='{h - 2}' font-size='10' fill='#5b6573' text-anchor='end'>max {hi:.1f}{unit}</text></svg>")


def show_feature(m, title, props, ex=None, labels=None):
    ex = {"what": (ex or {}).get("what", "")} if ex else None           # short popups: the side-panel card has the full guide
    body = _info_html(title, ex, feature_rows(props, labels, limit=14))
    extra = "".join(v for k, v in props.items() if k.endswith("_html") and isinstance(v, str))
    sp = props.get("spark")
    if isinstance(sp, str) and sp.startswith("["):
        import json as _json
        sp = _json.loads(sp.replace("None", "null").replace("nan", "null"))
    if isinstance(sp, list):
        extra += sparkline_svg(sp, f"Air temperature, last 7 days (from {props.get('spark_from', '')})", " °C")
    if extra:
        body = body[:-len("</div>")] + extra + "</div>"
    show_info(m, body)


def bind_info(m, gj: L.GeoJSON, title, ex=None, labels=None, title_field=None, hover=None):
    """Hover and click on a GeoJSON layer fill the info box (hover only for small layers, see HOVER_MAX_FEATURES)."""
    if hover is None:
        hover = len((gj.data or {}).get("features", [])) <= HOVER_MAX_FEATURES
    def handler(feature=None, **kw):
        if not feature:
            return
        p = feature.get("properties", {}) or {}
        t = f"{title}: {p.get(title_field)}" if title_field and p.get(title_field) else title
        show_feature(m, t, p, ex, labels)

    def click(feature=None, **kw):
        handler(feature, **kw)
        m._feature_click = time.time()
    if hover:
        gj.on_hover(handler)
    gj.on_click(click)
    return gj


@lru_cache(maxsize=64)
def _cog(layer_id):
    return rasterio.open(LY.LAYER_DIR / f"{layer_id}.tif")


def sample_raster(layer_id, lat, lon):
    try:
        ds = _cog(layer_id)
        x, y = TO3857.transform(lon, lat)
        v = float(next(ds.sample([(x, y)]))[0])
        return None if not np.isfinite(v) else v
    except Exception:
        return None


def _raster_click(m, **kw):
    if kw.get("type") != "click":
        return
    lat, lon = kw["coordinates"]
    shown = [(lid, w) for lid, w in getattr(m, "_rasters", {}).items() if w in m.layers and getattr(w, "visible", True)]
    if not shown:
        return
    blocks = []
    for lid, _w in shown[::-1][:3]:
        layer = LY.BY_ID.get(lid)
        v = sample_raster(lid, lat, lon)
        ex = explain.about(lid)
        if v is None:
            val = "no data here"
        elif layer and layer.categories:
            val = layer.categories.get(int(round(v)), "none")
        else:
            val = f"{v:,.2f} {ex.get('unit', '')}".strip()
        short = {"what": ex.get("what", "").split(":")[0], "when": ex.get("when", "")}
        blocks.append(_info_html(layer.title if layer else lid, short, value=val))
    body = f"<div style='font:11px system-ui;color:#5b6573'>at {lat:.5f}, {lon:.5f}</div>" + "<hr style='margin:4px 0'>".join(blocks)
    if time.time() - getattr(m, "_feature_click", 0) < 0.6 and m._info is not None:     # a feature was clicked too: keep it, add values
        body = m._info.value + "<hr style='margin:6px 0;border-top:2px solid #1f2a37'>" + body
    show_info(m, body)


# ------------------------------------------------------------------------------------------------------------ the map
def new_map(center=LY.ATHENS_CENTER, zoom=13, height="calc(100vh - 140px)", fit_municipality=False, basemap="Light grey"):
    m = leafmap.Map(center=list(center), zoom=zoom, height=height, max_zoom=19, min_zoom=11, draw_control=False, measure_control=False,
                    toolbar_control=False, search_control=False, layers_control=False, fullscreen_control=True, scale_control=True,
                    prefer_canvas=True)                      # canvas, not SVG: thousands of shapes pan smoothly
    m.prefer_canvas = True
    try:
        m.clear_layers()
    except Exception:
        pass
    m._lock = threading.RLock()
    m._rasters = {}
    m._base = []
    set_basemap(m, basemap)
    mask, outline, bounds = municipality_geojson()
    m._frame = [L.GeoJSON(data=mask, style={"fillColor": "#1f2a37", "fillOpacity": 0.38, "weight": 0}, name="outside Athens"),
                L.GeoJSON(data=outline, style={"color": "#1f2a37", "weight": 2.2, "opacity": 0.9, "fill": False}, name="municipality")]
    for f in m._frame:
        m.add_layer(f)
    m.observe(lambda ch: raise_frame(m), names="layers")
    # basemap switcher
    dd = W.Dropdown(options=list(BASEMAPS), value=basemap, layout=W.Layout(width="130px"))
    dd.observe(lambda ch: set_basemap(m, ch["new"]), names="value")
    m.add_control(L.WidgetControl(widget=dd, position="bottomright"))
    # info box
    m._info = W.HTML(INFO_HINT, layout=W.Layout(width="310px", padding="4px"))
    m.add_control(L.WidgetControl(widget=m._info, position="topright"))
    m.on_interaction(lambda **kw: _raster_click(m, **kw))
    if fit_municipality:
        view_to(m, bounds)
    return m


@lru_cache(maxsize=16)
def focus_geojson(dks=()):
    """(mask outside the selection, outline of the selection, bounds) for a district selection; () = whole municipality."""
    if not dks:
        return municipality_geojson()
    import geopandas as gpd
    from shapely.geometry import box, mapping
    g = gpd.GeoSeries([overview.district_geom(dks)], crs=2100).to_crs(4326).iloc[0]
    mask = box(19.0, 34.5, 29.5, 42.0).difference(g)
    b = g.bounds
    return ({"type": "Feature", "properties": {}, "geometry": mapping(mask)},
            {"type": "Feature", "properties": {}, "geometry": mapping(g.boundary)}, [[b[1], b[0]], [b[3], b[2]]])


def set_focus(m, dks=(), fit=True):
    """Dim everything outside the selected districts, outline them and zoom to them."""
    mask, outline, bounds = focus_geojson(tuple(dks))
    if getattr(m, "_frame", None):
        m._frame[0].data = mask
        m._frame[1].data = outline
    if fit:
        view_to(m, bounds)


def view_to(m, bounds, px=(720, 480)):
    """Zoom to bounds by setting center/zoom traits (persisted state, so it also works before the map is drawn), then fit."""
    import math
    (s_, w_), (n_, e_) = bounds
    lat = (s_ + n_) / 2

    def merc(y):
        return math.log(math.tan(math.pi / 4 + math.radians(y) / 2))
    zx = math.log2(px[0] / 256 * 360 / max(e_ - w_, 1e-6))
    zy = math.log2(px[1] / 256 * 2 * math.pi / max(merc(n_) - merc(s_), 1e-9))
    z = max(11, min(17, math.floor(min(zx, zy))))
    c = [lat, (w_ + e_) / 2]

    def apply():
        m.center, m.zoom = c, z
        try:
            m.fit_bounds(bounds)
        except Exception:
            pass
    apply()
    # the browser may still be laying the map out (size 0) when the first command arrives; repeat once it is drawn
    m._focus_seq = getattr(m, "_focus_seq", 0) + 1
    seq = m._focus_seq

    try:
        from solara.server import kernel_context
        ctx = kernel_context.get_current_context()
    except Exception:
        ctx = None

    def again():
        if getattr(m, "_focus_seq", 0) != seq:
            return
        if ctx is not None:
            with ctx:
                apply()
        else:
            apply()
    for delay in (0.8, 2.0):
        t = threading.Timer(delay, again)
        t.daemon = True
        t.start()


def _rank(m, l):
    if l in getattr(m, "_frame", []):
        return 5
    if isinstance(l, L.TileLayer):
        return 0
    if isinstance(l, (L.ImageOverlay, L.Heatmap)):
        return 1
    if isinstance(l, L.GeoJSON):
        f = (l.data or {}).get("features") or [{}]
        return 3 if (f[0].get("geometry") or {}).get("type") in ("Point", "MultiPoint") else 2
    return 4


def raise_frame(m):
    """Layer order: basemap, rasters, polygons/lines, points, animations, then the outside-Athens mask and outline on top."""
    ls = list(m.layers)
    order = sorted(ls, key=lambda l: _rank(m, l))              # stable: keeps the add order within each rank
    if order == ls:
        return
    with m._lock:
        m.layers = tuple(order)


def set_basemap(m, name):
    with m._lock:
        new = [L.TileLayer(name=f"{name} {i}", base=(i == 0), max_zoom=19, **spec) for i, spec in enumerate(BASEMAPS[name])]
        rest = [l for l in m.layers if l not in m._base]
        old_tiles = [l for l in rest if isinstance(l, L.TileLayer)]
        m.layers = tuple(new + [l for l in rest if l not in old_tiles])
        m._base = new


def build_layer(layer_id, m=None, detail="Standard") -> L.Layer:
    """ipyleaflet layer for a catalogue layer; if a map is given, hover/click fill its info box."""
    layer = LY.get(layer_id)
    ex = explain.about(layer_id)
    if layer.kind == "raster":
        meta = overview.raster_meta(layer_id)
        w = L.ImageOverlay(url=_png_url(layer_id), bounds=meta["bounds"], name=layer.title, opacity=0.7)
        if m is not None:
            m._rasters[layer_id] = w
        return w
    if layer.kind == "heat":
        return L.Heatmap(locations=_heat(layer_id), radius=9, blur=12, max_zoom=16, name=layer.title,
                         gradient={0.2: "#ffffb2", 0.5: "#fd8d3c", 0.8: "#e31a1c", 1.0: "#800026"})
    data = _geojson(layer_id, DETAIL_TOL.get(detail) if layer.kind != "point" else None, DETAIL_DIGITS.get(detail, 5))
    st = dict(layer.style)
    chor = st.pop("choropleth", None)
    cmap = st.pop("cmap", "YlOrRd")
    cat = st.pop("categorical", None)
    vr = st.pop("vrange", None)
    if layer.kind == "point" and chor:
        vals = [f["properties"].get(chor) for f in data["features"] if f["properties"].get(chor) is not None]
        lo, hi = vr or ((min(vals), max(vals)) if vals else (0, 1))

        def pstyle(feature, lo=lo, hi=hi):
            v = feature["properties"].get(chor)
            return {"radius": st.get("radius", 6), "color": st.get("color", "#333"), "weight": st.get("weight", 1),
                    "fillOpacity": st.get("fillOpacity", 0.9), "fillColor": "#d0d5dd" if v is None else colour(cmap, v, lo, hi)}
        gj = L.GeoJSON(data=data, name=layer.title, point_style={"radius": st.get("radius", 6)}, style_callback=pstyle)
    elif layer.kind == "point":
        gj = L.GeoJSON(data=data, name=layer.title, point_style={"radius": st.get("radius", 4), "color": st.get("color", "#333"),
                                                                 "fillColor": st.get("fillColor", "#999"), "fillOpacity": st.get("fillOpacity", 0.8),
                                                                 "weight": st.get("weight", 1)})
    elif cat:
        pal = category_palette(layer_id)

        def cstyle(feature):
            return {"fillColor": pal.get(feature["properties"].get(cat), "#d0d5dd"), "color": st.get("color", "#333"),
                    "weight": st.get("weight", 0.5), "fillOpacity": st.get("fillOpacity", 0.55)}
        gj = L.GeoJSON(data=data, name=layer.title, style_callback=cstyle, hover_style={"weight": 2.5, "color": "#000"})
    elif chor:
        vals = [f["properties"].get(chor) for f in data["features"] if f["properties"].get(chor) is not None]
        lo, hi = (min(vals), max(vals)) if vals else (0, 1)

        def style_cb(feature, lo=lo, hi=hi):
            v = feature["properties"].get(chor)
            return {"fillColor": "#cccccc" if v is None else colour(cmap, v, lo, hi), "color": st.get("color", "#333"),
                    "weight": st.get("weight", 0.5), "fillOpacity": st.get("fillOpacity", 0.6)}
        gj = L.GeoJSON(data=data, name=layer.title, style_callback=style_cb, hover_style={"weight": 2.5, "color": "#000"})
    else:
        gj = L.GeoJSON(data=data, name=layer.title, style=st, hover_style={"weight": st.get("weight", 1) + 1.5, "color": "#000"})
    if m is not None:
        keys = set(layer.keep or ()) or set((data.get("features") or [{}])[0].get("properties", {}))
        tf = next((f for f in ("neighbourhood_name_en", "name_en", "name", "synoikia", "street_gr", "odos") if f in keys), None)
        bind_info(m, gj, layer.title, ex, title_field=tf)
    return gj


def category_palette(layer_id):
    if layer_id == "area_types":
        from ..services import measures
        return measures.TYPE_COLOURS
    return {}


def legend_items(layer_id):
    """(label, colour) pairs for a small legend."""
    layer = LY.get(layer_id)
    if layer.style.get("categorical"):
        return list(category_palette(layer_id).items())
    if layer.kind == "point" and layer.style.get("choropleth"):
        lo, hi = layer.style.get("vrange", (0, 1))
        unit = explain.about(layer_id).get("unit", "")
        return [(f"{lo + (hi - lo) * t:.3g} {unit}".strip(), colour(layer.style.get("cmap", "YlOrRd"), t)) for t in (0, 0.5, 1)]
    if layer.kind == "raster":
        meta = overview.raster_meta(layer_id)
        if layer.categories:
            n = max(len(layer.categories), 1)
            return [(lab, colour(layer.cmap, 0.85 if layer.vrange == (0, 1) else i / max(n - 1, 1))) for i, lab in enumerate(layer.categories.values())]
        lo, hi = meta["vmin"], meta["vmax"]
        unit = explain.about(layer_id).get("unit", "")
        return [(f"{lo + (hi - lo) * t:.3g} {unit}".strip(), colour(layer.cmap, t)) for t in (0, 0.5, 1)]
    st = layer.style
    return [(layer.title, st.get("fillColor") or st.get("color", "#666"))]


def index_card_html(layer_id, open_more=False):
    """Compact, consistent card for any raster or coloured layer: what it is, a colour bar with plain-language ends,
    what it stands for, when; method/source behind 'More'."""
    e = explain.about(layer_id)
    layer = LY.BY_ID.get(layer_id)
    title = layer.title if layer else layer_id
    lo_lab, hi_lab = e.get("scale", ("", ""))
    bar = ""
    if layer is not None and (layer.kind == "raster" and not layer.categories or layer.style.get("choropleth")):
        cmap = layer.cmap if layer.kind == "raster" else layer.style.get("cmap", "YlOrRd")
        if layer.kind == "raster":
            meta = overview.raster_meta(layer_id)
            lo, hi = meta["vmin"], meta["vmax"]
        else:
            lo, hi = layer.style.get("vrange", (None, None))
        stops = ", ".join(colour(cmap, t) for t in np.linspace(0, 1, 7))
        unit = e.get("unit", "")
        num = (f"<span>{lo:.3g}{(' ' + unit) if unit else ''}</span><span>{hi:.3g}{(' ' + unit) if unit else ''}</span>"
               if lo is not None else "")
        bar = (f"<div style='height:10px;border-radius:5px;margin:6px 0 2px;background:linear-gradient(90deg,{stops})'></div>"
               f"<div style='display:flex;justify-content:space-between;font-size:11px;color:#344054'>{num}</div>"
               + (f"<div style='display:flex;justify-content:space-between;font-size:11px;color:#5b6573'><span>{html.escape(lo_lab)}</span>"
                  f"<span style='text-align:right'>{html.escape(hi_lab)}</span></div>" if lo_lab or hi_lab else ""))
    elif layer is not None and (layer.style.get("categorical") or layer.categories):
        bar = "<div style='margin:4px 0'>" + " ".join(
            f"<span style='display:inline-block;margin:1px 6px 1px 0;font-size:11px'><span style='display:inline-block;width:10px;"
            f"height:10px;border-radius:2px;background:{c};vertical-align:-1px;margin-right:3px'></span>{html.escape(l)}</span>"
            for l, c in legend_items(layer_id)) + "</div>"
    when = (f"<span style='display:inline-block;background:#eef2f6;color:#344054;border-radius:10px;padding:1px 8px;font-size:11px;"
            f"margin-top:4px'>{html.escape(e['when'])}</span>") if e.get("when") else ""
    more = "".join(f"<div style='margin:2px 0'><b>{k}:</b> {html.escape(e[f])}</div>"
                   for k, f in (("Stands for", "proxy"), ("How it's made", "method"), ("Source", "source")) if e.get(f))
    return ("<div style='border:1px solid #e6e8eb;border-radius:8px;padding:8px 10px;margin:6px 0;background:#fff;font-size:12.5px;"
            "line-height:1.4;color:#1f2a37'>"
            f"<div style='font-weight:700'>{html.escape(title)}</div>"
            f"<div style='color:#344054'>{html.escape(e.get('what', ''))}</div>{bar}"
            + (f"<div style='margin-top:4px'><b>How to read:</b> {html.escape(e['read'])}</div>" if e.get("read") else "")
            + when
            + (f"<details {'open' if open_more else ''} style='margin-top:4px'><summary style='cursor:pointer;color:#5b6573'>More</summary>"
               f"<div style='color:#344054;font-size:12px'>{more}</div></details>" if more else "")
            + "</div>")


def explain_html(layer_id, short=False):
    """Plain-language explanation block for legends / side panels."""
    e = explain.about(layer_id)
    t = LY.BY_ID[layer_id].title if layer_id in LY.BY_ID else layer_id
    bits = [f"<b>{html.escape(t)}</b>: {html.escape(e.get('what', ''))}"]
    if e.get("proxy") and not short:
        bits.append(f"<i>Stands for:</i> {html.escape(e['proxy'])}")
    if e.get("read"):
        bits.append(f"<i>How to read:</i> {html.escape(e['read'])}")
    if e.get("when"):
        bits.append(f"<i>When:</i> {html.escape(e['when'])}")
    return "<div class='note' style='margin:4px 0 8px'>" + "<br>".join(bits) + "</div>"
