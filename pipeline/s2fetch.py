"""Live data access: Sentinel-2 scenes, ESA WorldCover and cached composites, all read straight onto the analysis grid.

Everything is read with a WarpedVRT onto the 10 m analysis grid, so only the blocks inside the area of interest are downloaded and any
CRS or resolution difference (20 m SWIR bands, UTM zone changes) is handled by GDAL. Per scene and band the masked reflectance is cached
as .npy, so changing the index list, the classifier or the date window never re-downloads what is already on disk.
The STAC functions need network access and the pystac-client and planetary-computer packages; they are NOT testable offline here."""
import hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, shape

from . import indices as ix, rasters
from .scaling import resolve_scale_offset

STAC_PC = "https://planetarycomputer.microsoft.com/api/stac/v1"
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.vrt",
                GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="1", VSI_CACHE="TRUE")
BAD_SCL = (0, 1, 3, 6, 7, 8, 9, 10, 11)       # no data, saturated, cloud shadow, cloud (med/high), cirrus, snow


def grid_key(grid):
    t = grid["transform"]
    return hashlib.sha1(json.dumps([str(grid["crs"]), list(t)[:6], grid["height"], grid["width"]]).encode()).hexdigest()[:12]


def grid_bbox_wgs84(grid):
    from rasterio.warp import transform_bounds
    t = grid["transform"]
    b = (t.c, t.f + t.e * grid["height"], t.c + t.a * grid["width"], t.f)
    return transform_bounds(grid["crs"], "EPSG:4326", *b)


def warp_read(href, grid, resampling=Resampling.bilinear, band=1, src_nodata=None, nodata=None):
    """Read one band of a (possibly remote) raster onto the grid. Works for local files and https/vsicurl hrefs."""
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            kw = dict(crs=grid["crs"], transform=grid["transform"], width=grid["width"], height=grid["height"], resampling=resampling)
            if src_nodata is not None or src.nodata is not None:
                kw["src_nodata"] = src.nodata if src_nodata is None else src_nodata
                kw["nodata"] = kw["src_nodata"] if nodata is None else nodata
            with WarpedVRT(src, **kw) as vrt:
                return vrt.read(band)


# --- search ----------------------------------------------------------------------------------------------------------------
def search_scenes(bbox, live, client=None):
    """STAC search with cloud, month and tile filters. Returns items (signed for Planetary Computer), sorted by date.
    Only one MGRS tile is used (the one covering most of the area) so the area never needs mosaicking."""
    if client is None:
        from pystac_client import Client
        import planetary_computer as pc
        client = Client.open(live.get("stac_url", STAC_PC), modifier=pc.sign_inplace)
    start, end = live.get("start") or f"{live['year']}-06-01", live.get("end") or f"{live['year']}-09-30"
    items = list(client.search(collections=[live.get("collection", "sentinel-2-l2a")], bbox=list(bbox), datetime=f"{start}/{end}",
                               query={"eo:cloud_cover": {"lt": live.get("max_cloud_pct", 20)}}).items())
    months = set(live.get("months") or [])
    if months:
        items = [i for i in items if i.datetime.month in months]
    if live.get("tile"):
        items = [i for i in items if i.properties.get("s2:mgrs_tile") == live["tile"]]
    if not items:
        return []
    aoi = box(*bbox)
    cover = {}
    for i in items:
        t = i.properties.get("s2:mgrs_tile", "?")
        cover.setdefault(t, []).append(shape(i.geometry).intersection(aoi).area / aoi.area)
    best = max(cover, key=lambda t: (np.mean(cover[t]), len(cover[t])))
    items = [i for i in items if i.properties.get("s2:mgrs_tile", "?") == best]
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    items = items[: live.get("max_scenes", 8)]
    return sorted(items, key=lambda i: i.datetime)


def scene_table(items):
    return [{"id": i.id, "datetime": i.datetime.isoformat(), "cloud_pct": i.properties.get("eo:cloud_cover"),
             "tile": i.properties.get("s2:mgrs_tile")} for i in items]


# --- scenes -> masked reflectance ------------------------------------------------------------------------------------------
def scene_bands(item, bands, grid, cache_dir=None, log=print):
    """Masked reflectance (NaN where cloud, shadow, snow, no data) for the requested bands of one scene, on the grid.
    Bands already on disk are not downloaded again."""
    cdir = Path(cache_dir) / "s2" / grid_key(grid) if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)

    def path(b):
        return cdir / f"{item.id}_{b}.npy" if cdir else None

    def cached(b):
        p = path(b)
        return np.load(p) if p is not None and p.exists() else None

    out = {b: cached(b) for b in bands}
    todo = [b for b, v in out.items() if v is None]
    if todo:
        scl = cached("SCL")

        def read(b):
            if b == "SCL":
                return b, warp_read(item.assets["SCL"].href, grid, Resampling.nearest)
            a = item.assets[b]
            sc, of, _ = resolve_scale_offset(a.href, a.extra_fields, item.properties)
            dn = warp_read(a.href, grid, Resampling.bilinear, src_nodata=0, nodata=0)
            return b, np.where(dn > 0, dn.astype("float32") * sc + of, np.nan).astype("float32")

        with ThreadPoolExecutor(max_workers=4) as ex:
            got = dict(ex.map(read, todo + (["SCL"] if scl is None else [])))
        if scl is None:
            scl = got.pop("SCL")
            if cdir:
                np.save(path("SCL"), scl)
        bad = np.isin(scl, BAD_SCL)
        for b in todo:
            a = got[b]
            a[bad] = np.nan
            out[b] = a
            if cdir:
                np.save(path(b), a)
    return out


def composite_indices(items, names, grid, min_valid, cache_dir=None, log=print):
    """Median composite of each index over the scenes. Returns (dict name -> array, n_valid_scenes array)."""
    need = [b for b in ix.bands_needed(names) if b != "SCL"]
    per = {n: [] for n in names}
    for k, it in enumerate(items, 1):
        log(f"  scene {k}/{len(items)} {it.id}")
        bands = scene_bands(it, need, grid, cache_dir, log)
        for n, v in ix.compute(names, bands).items():
            per[n].append(v)
    out, nv = {}, None
    for n, lst in per.items():
        out[n], nv_ = rasters.composite(np.stack(lst), min_valid)
        nv = nv_ if nv is None else nv
    return out, nv


# --- reference land cover ---------------------------------------------------------------------------------------------------
def fetch_worldcover(bbox, grid, year=2021, client=None, log=print):
    """ESA WorldCover (10 m) codes on the grid; 0 = no data. 2020 (v100) and 2021 (v200) exist."""
    if client is None:
        from pystac_client import Client
        import planetary_computer as pc
        client = Client.open(STAC_PC, modifier=pc.sign_inplace)
    items = list(client.search(collections=["esa-worldcover"], bbox=list(bbox), datetime=f"{year}-01-01/{year}-12-31").items())
    if not items:
        raise RuntimeError(f"no WorldCover tiles for {year} over the area")
    out = np.zeros((grid["height"], grid["width"]), "uint8")
    for it in items:
        a = warp_read(it.assets["map"].href, grid, Resampling.nearest)
        out = np.where(out == 0, a, out)
    return out
