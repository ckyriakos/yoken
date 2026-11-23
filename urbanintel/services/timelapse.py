"""Satellite timelapses for Athens or selected districts, built from freshly downloaded scenes.

For each time step (every year in a season, or every month) the clearest Sentinel-2 or Landsat scenes are fetched from the
Microsoft Planetary Computer STAC, cloud-masked, combined into a median composite, and turned into an index (NDVI, NDWI, MNDWI,
NDBI, Landsat surface temperature) or a true-colour image. Each step becomes a titled frame of an animated GIF, a map overlay and
one point of a trend line (mean value inside the area). Like leafmap's landsat_timelapse, but with the platform's indices.

Data are read on a Web-Mercator (EPSG:3857) grid so overlays sit exactly on the web map. Scene bands are cached as .npy, so a second
timelapse over the same area and years only downloads what is new. Results: data/outputs/timelapse/<key>/."""
from __future__ import annotations

import calendar
import hashlib
import io
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from pipeline import indices as IX
from pipeline import s2fetch
from pipeline.scaling import resolve_scale_offset
from .. import layers as LY
from . import overview

OUT = LY.OUT / "timelapse"
CACHE = LY.ROOT / "data" / "cache"

SENSORS = {
    "sentinel-2": {"label": "Sentinel-2", "first_year": 2017, "res_m": 20, "credit": "Contains modified Copernicus Sentinel data"},
    "landsat": {"label": "Landsat 5/7/8/9", "first_year": 1984, "res_m": 30, "credit": "Landsat imagery courtesy of NASA/USGS"},
}
PRODUCTS = {   # id: (label, colormap, value range, unit, sensors)
    "true_color": ("True colour", None, None, "", ("sentinel-2", "landsat")),
    "ndvi": ("Greenness (NDVI)", "RdYlGn", (-0.1, 0.7), "", ("sentinel-2", "landsat")),
    "ndwi": ("Water index (NDWI)", "RdBu", (-0.6, 0.2), "", ("sentinel-2", "landsat")),
    "mndwi": ("Water index (MNDWI)", "RdBu", (-0.6, 0.2), "", ("sentinel-2", "landsat")),
    "ndbi": ("Built-up index (NDBI)", "OrRd", (-0.3, 0.3), "", ("sentinel-2", "landsat")),
    "lst": ("Surface temperature", "inferno", (25, 55), "°C", ("landsat",)),
}
SEASONS = {"Summer (Jun–Sep)": (6, 9), "Wet season (Jan–Apr)": (1, 4), "Autumn (Oct–Dec)": (10, 12), "Whole year": (1, 12)}
# Landsat on the Planetary Computer: S2-style band names so pipeline.indices works unchanged
LANDSAT_BANDS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir08", "B11": "swir16"}
LANDSAT_BAD_QA = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)    # fill, dilated cloud, cloud, shadow, snow
MAX_STEPS = 16


# ------------------------------------------------------------------------------------------------------------ setup
def area(dks=()):
    """(polygon EPSG:2100, label) for the selected districts or the whole municipality."""
    dks = overview.norm_districts(dks)
    lab = "Athens" if not dks else "Athens · " + ", ".join(overview.ORD[d] for d in dks) + (" district" if len(dks) == 1 else " districts")
    return overview.district_geom(dks), lab


def grid_3857(poly2100, res_m, margin_m=300):
    """Web-Mercator grid covering the area. 3857 metres are stretched by 1/cos(lat), so the cell size is scaled to stay ~res_m."""
    b4326 = gpd.GeoSeries([poly2100.buffer(margin_m)], crs=2100).to_crs(4326).total_bounds
    lat = (b4326[1] + b4326[3]) / 2
    res = res_m / np.cos(np.radians(lat))
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", "EPSG:3857", *b4326)
    w, h = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    return {"crs": "EPSG:3857", "transform": from_origin(x0, y1, res, res), "width": w, "height": h}, tuple(b4326)


def windows(start_year, end_year, months=(6, 9), step="yearly"):
    """[(label, start, end)] time steps. yearly: the month span of every year; monthly: every month from start to end year."""
    m0, m1 = months
    if m1 < m0:
        raise ValueError("the season must not cross the new year")
    out = []
    if step == "yearly":
        for y in range(int(start_year), int(end_year) + 1):
            out.append((str(y) if (m0, m1) == (1, 12) else f"{calendar.month_abbr[m0]}–{calendar.month_abbr[m1]} {y}",
                        date(y, m0, 1), date(y, m1, calendar.monthrange(y, m1)[1])))
    else:
        for y in range(int(start_year), int(end_year) + 1):
            for mo in range(m0, m1 + 1):
                out.append((f"{calendar.month_abbr[mo]} {y}", date(y, mo, 1), date(y, mo, calendar.monthrange(y, mo)[1])))
    today = date.today()
    out = [w for w in out if w[1] <= today]
    if len(out) > MAX_STEPS:
        raise ValueError(f"{len(out)} time steps requested; the limit is {MAX_STEPS} (shorten the period or use yearly steps)")
    return out


def key(**p):
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------------ scenes
def _client():
    from pystac_client import Client
    import planetary_computer as pc
    return Client.open(s2fetch.STAC_PC, modifier=pc.sign_inplace)


def search(sensor, bbox, start, end, max_cloud, max_scenes, client=None):
    client = client or _client()
    if sensor == "sentinel-2":
        return s2fetch.search_scenes(bbox, {"start": f"{start}", "end": f"{end}", "max_cloud_pct": max_cloud, "max_scenes": max_scenes},
                                     client=client)
    items = list(client.search(collections=["landsat-c2-l2"], bbox=list(bbox), datetime=f"{start}/{end}",
                               query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    ok = []
    for it in items:
        plat = it.properties.get("platform", "")
        if plat not in ("landsat-5", "landsat-7", "landsat-8", "landsat-9"):
            continue
        if plat == "landsat-7" and it.datetime.date() > date(2003, 5, 31):    # scan-line corrector failure: striped images
            continue
        ok.append(it)
    ok.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return sorted(ok[:max_scenes], key=lambda i: i.datetime)


def _landsat_bands(item, need, grid):
    """Masked Landsat surface reflectance (S2 band names) and surface temperature (°C) on the grid, cached per scene."""
    cdir = CACHE / "landsat" / s2fetch.grid_key(grid)
    cdir.mkdir(parents=True, exist_ok=True)
    out, qa = {}, None
    for b in need:
        p = cdir / f"{item.id}_{b}.npy"
        if p.exists():
            out[b] = np.load(p)
            continue
        if qa is None:
            qp = cdir / f"{item.id}_QA.npy"
            if qp.exists():
                qa = np.load(qp)
            else:
                qa = s2fetch.warp_read(item.assets["qa_pixel"].href, grid, Resampling.nearest)
                np.save(qp, qa)
        if b == "LST":
            asset = "lwir11" if "lwir11" in item.assets else "lwir"
            fb = (0.00341802, 149.0)
        else:
            asset, fb = LANDSAT_BANDS[b], (0.0000275, -0.2)
        a = item.assets[asset]
        sc, of, _ = resolve_scale_offset(a.href, a.extra_fields, item.properties, fallback=fb)
        dn = s2fetch.warp_read(a.href, grid, Resampling.bilinear, src_nodata=0, nodata=0).astype("float32")
        v = np.where(dn > 0, dn * sc + of, np.nan).astype("float32")
        if b == "LST":
            v = v - 273.15
        v[(qa.astype("int64") & LANDSAT_BAD_QA) != 0] = np.nan
        np.save(p, v)
        out[b] = v
    return out


def composite(sensor, items, product, grid, log=print):
    """Median composite of the product over the scenes. Returns (2-D array or RGB stack, number of clear observations)."""
    if product == "true_color":
        need = ["B04", "B03", "B02"]
    elif product == "lst":
        need = ["LST"]
    else:
        need = [b for b in IX.bands_needed([product]) if b != "SCL"]
    stacks = {b: [] for b in need}
    for k, it in enumerate(items, 1):
        log(f"    scene {k}/{len(items)} {it.id}")
        bands = s2fetch.scene_bands(it, need, grid, CACHE) if sensor == "sentinel-2" else _landsat_bands(it, need, grid)
        for b in need:
            stacks[b].append(bands[b])
    with np.errstate(all="ignore"):
        med = {b: np.nanmedian(np.stack(v), axis=0) for b, v in stacks.items()}
        nclear = np.sum(np.isfinite(np.stack(stacks[need[0]])), axis=0)
    if product == "true_color":
        return np.stack([med["B04"], med["B03"], med["B02"]]), nclear
    if product == "lst":
        return med["LST"], nclear
    return IX.compute([product], med)[product], nclear


# ------------------------------------------------------------------------------------------------------------ rendering
def _rgb(stack):
    """Reflectance -> display RGB (fixed stretch, so frames are comparable)."""
    with np.errstate(all="ignore"):
        rgb = np.clip(np.moveaxis(stack, 0, -1) * 3.2, 0, 1) ** (1 / 1.25)
    alpha = np.all(np.isfinite(rgb), axis=-1)
    return np.nan_to_num(rgb), alpha


def overlay_png(arr, product, path):
    """Colourised RGBA PNG (no decorations) for the web map."""
    from PIL import Image
    if product == "true_color":
        rgb, a = _rgb(arr)
        rgba = np.dstack([(rgb * 255).astype("uint8"), np.where(a, 235, 0).astype("uint8")])
    else:
        lab, cmap, vr, unit, _ = PRODUCTS[product]
        lyr = LY.Layer("tl", lab, "", "", "raster", cmap=cmap, vrange=vr)
        rgba, *_ = overview.colourise(arr, lyr, alpha=215)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)


def frame_png(arr, product, extent3857, outline3857, title, subtitle, credit, path, inside=None):
    """Titled frame for the animation: image, area outline, colour bar, date, credit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = (arr.shape[1:] if product == "true_color" else arr.shape)
    fig_w = 7.2
    fig = plt.figure(figsize=(fig_w, fig_w * h / w * 0.92 + 0.9), dpi=110)
    ax = fig.add_axes([0.02, 0.08, 0.84 if product != "true_color" else 0.96, 0.78])
    if product == "true_color":
        rgb, a = _rgb(arr)
        im = ax.imshow(np.dstack([rgb, a.astype(float)]), extent=extent3857, interpolation="nearest")
    else:
        lab, cmap, vr, unit, _ = PRODUCTS[product]
        im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vr[0], vmax=vr[1], extent=extent3857, interpolation="nearest")
        cax = fig.add_axes([0.88, 0.12, 0.025, 0.68])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"{lab} {('(' + unit + ')') if unit else ''}".strip(), fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if inside is not None:                       # dim everything outside the selected area
        veil = np.dstack([np.ones(inside.shape)] * 3 + [np.where(inside, 0.0, 0.5)])     # white, 50% outside the area
        ax.imshow(veil, extent=extent3857, interpolation="nearest")
    outline3857.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=1.6)
    outline3857.plot(ax=ax, facecolor="none", edgecolor="#111", linewidth=0.7)
    ax.set_axis_off()
    fig.text(0.02, 0.955, title, fontsize=12, fontweight="bold", color="#1f2a37", va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=10, color="#344054", va="top")
    fig.text(0.02, 0.03, credit, fontsize=7, color="#5b6573")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------------ main
def build(sensor="sentinel-2", product="ndvi", start_year=2017, end_year=2026, season="Summer (Jun–Sep)", step="yearly",
          districts=(), max_cloud_pct=20, max_scenes=3, frame_ms=900, force=False, log=print, progress=None):
    """Download scenes and build the timelapse. Returns metadata (frames, GIF path, trend). Cached by parameters."""
    if sensor not in SENSORS:
        raise ValueError(f"sensor must be one of {list(SENSORS)}")
    if product not in PRODUCTS or sensor not in PRODUCTS[product][4]:
        raise ValueError(f"{product} is not available for {sensor}")
    start_year = max(int(start_year), SENSORS[sensor]["first_year"])
    months = SEASONS.get(season, season) if isinstance(season, str) else tuple(season)
    wins = windows(start_year, end_year, months, step)
    if not wins:
        raise ValueError("no time steps in the requested period")
    dks = overview.norm_districts(districts)
    params = dict(sensor=sensor, product=product, start_year=start_year, end_year=int(end_year), months=list(months), step=step,
                  districts=list(dks), max_cloud_pct=int(max_cloud_pct), max_scenes=int(max_scenes))
    k = key(**params)
    d = OUT / k
    if (d / "meta.json").exists() and not force:
        m = json.loads((d / "meta.json").read_text())
        if m.get("status") == "done":
            return m
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "map").mkdir(parents=True, exist_ok=True)
    prog = progress or (lambda f, msg: None)
    poly, aoi_label = area(dks)
    grid, bbox = grid_3857(poly, SENSORS[sensor]["res_m"])
    t = grid["transform"]
    extent = (t.c, t.c + t.a * grid["width"], t.f + t.e * grid["height"], t.f)
    outline = gpd.GeoSeries([poly], crs=2100).to_crs(3857)
    inside = rasterize([(outline.iloc[0], 1)], out_shape=(grid["height"], grid["width"]), transform=t, fill=0, dtype="uint8").astype(bool)
    lab, cmap, vr, unit, _ = PRODUCTS[product]
    meta = {"key": k, "status": "running", "params": params, "area": aoi_label, "product_label": lab, "unit": unit,
            "sensor_label": SENSORS[sensor]["label"], "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]], "frames": [], "skipped": [],
            "frame_ms": frame_ms, "created": pd.Timestamp.now().isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    client = _client()
    try:
        for i, (wl, w0, w1) in enumerate(wins):
            prog(i / len(wins), f"{wl}: searching scenes ({i + 1} of {len(wins)})")
            items = search(sensor, bbox, w0, w1, max_cloud_pct, max_scenes, client)
            if not items:
                meta["skipped"].append({"label": wl, "reason": f"no scenes under {max_cloud_pct}% cloud"})
                log(f"  {wl}: no scenes")
                continue
            prog((i + 0.3) / len(wins), f"{wl}: downloading {len(items)} image(s) ({i + 1} of {len(wins)})")
            arr, nclear = composite(sensor, items, product, grid, log)
            valid = (np.all(np.isfinite(arr), axis=0) if product == "true_color" else np.isfinite(arr)) & inside
            cover = float(valid.sum() / max(inside.sum(), 1))
            if cover < 0.5:
                meta["skipped"].append({"label": wl, "reason": f"only {cover:.0%} of the area cloud-free"})
                continue
            mean = None if product == "true_color" else float(np.nanmean(np.where(inside, arr, np.nan)))
            stem = f"{i:02d}"
            sub = f"{wl} · {lab} · {SENSORS[sensor]['label']}, median of {len(items)} image(s)"
            frame_png(arr, product, extent, outline, aoi_label, sub,
                      f"{SENSORS[sensor]['credit']} · Microsoft Planetary Computer · Athens Urban Intelligence", d / "frames" / f"{stem}.png",
                      inside=inside)
            overlay_png(arr, product, d / "map" / f"{stem}.png")
            meta["frames"].append({"i": i, "label": wl, "start": str(w0), "end": str(w1), "n_images": len(items), "mean": mean,
                                   "cover": round(cover, 3), "dates": sorted({it.datetime.strftime("%Y-%m-%d") for it in items}),
                                   "scenes": [it.id for it in items], "frame": f"frames/{stem}.png", "overlay": f"map/{stem}.png"})
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
        if not meta["frames"]:
            raise RuntimeError("no usable images in any time step; raise the cloud limit or change the season")
        prog(0.97, "assembling the animation")
        _gif(d, meta, frame_ms)
        meta["status"] = "done"
        meta["gif"] = "timelapse.gif"
    except Exception as e:
        meta["status"] = "error"
        meta["error"] = str(e)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        raise
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    prog(1.0, "done")
    return meta


def _gif(d, meta, frame_ms):
    from PIL import Image
    frames = [Image.open(d / f["frame"]).convert("RGB") for f in meta["frames"]]
    pal = [f.quantize(colors=255, method=Image.Quantize.MEDIANCUT) for f in frames]
    dur = [frame_ms] * len(pal)
    dur[-1] = frame_ms * 2                       # pause on the last frame
    pal[0].save(d / "timelapse.gif", save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)


def get(k):
    p = OUT / k / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def listing():
    if not OUT.exists():
        return []
    out = []
    for p in OUT.glob("*/meta.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda m: m.get("created", ""), reverse=True)


def path(k, rel):
    p = (OUT / k / rel).resolve()
    if not str(p).startswith(str((OUT / k).resolve())):
        raise ValueError("bad path")
    return p
