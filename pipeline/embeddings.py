"""Geospatial embeddings as land-cover features: Google/DeepMind AlphaEarth Foundations 'Satellite Embedding' V1 (annual, 10 m,
64 axes A00..A63, 2017-2025, CC-BY 4.0), openly hosted as cloud-optimised GeoTIFFs by Source Cooperative (tge-labs/aef).
Values are int8 (-127..127, -128 = masked) and are de-quantised as sign(v) * (v / 127.5)**2. They must be resampled on the RAW integers
with nearest neighbour (averaging quantised values is wrong), which is what we do: the tiles are 10 m and only a few metres off our grid.
The tiles are stored bottom-up, so the VRT next to each .tiff is opened first. File layout, VRT naming and the manifest are taken from the
published README and have not been exercised against the live bucket from here: set `paths` in lulc.yaml to override discovery."""
import numpy as np
from rasterio.enums import Resampling
import rasterio
from shapely.geometry import box

from . import s2fetch

BASE_URL = "https://data.source.coop/tge-labs/aef/"
N_AXES = 64


def dequantize(raw):
    """int8 array (..., ) -> float32 in [-1, 1]; masked (-128) -> NaN."""
    v = raw.astype("float32")
    out = np.sign(v) * (v / 127.5) ** 2
    out[raw == -128] = np.nan
    return out


def utm_zones(bbox):
    lo, hi = int((bbox[0] + 180) // 6) + 1, int((bbox[2] + 180) // 6) + 1
    hemi = "N" if (bbox[1] + bbox[3]) / 2 >= 0 else "S"
    return [f"{z}{hemi}" for z in range(lo, hi + 1)]


def discover_tiles(bbox, year, base_url=BASE_URL, manifest_text=None, open_fn=None):
    """Hrefs of the tiles (VRT preferred) that intersect the bbox. manifest_text is the content of v1/annual/manifest.txt;
    when None it is downloaded. open_fn(href) -> (bounds in WGS84) lets tests avoid the network."""
    if manifest_text is None:
        import urllib.request
        manifest_text = urllib.request.urlopen(base_url + "v1/annual/manifest.txt", timeout=60).read().decode()
    zones = utm_zones(bbox)
    names = [ln.strip() for ln in manifest_text.splitlines() if ln.strip().endswith(".tiff")]
    cand = [n for n in names if f"/{year}/" in n and any(f"/{z}/" in n for z in zones)]
    aoi = box(*bbox)
    hits = []
    for n in cand:
        rel = n.split("v1/annual/", 1)[-1]
        href = f"{base_url}v1/annual/{rel}"
        vrt = href[:-5] + ".vrt"
        wb = (open_fn or _bounds_wgs84)(vrt)
        if wb is not None and box(*wb).intersects(aoi):
            hits.append(vrt)
    return hits


def _bounds_wgs84(href):
    from rasterio.warp import transform_bounds
    try:
        with rasterio.Env(**s2fetch.GDAL_ENV), rasterio.open(href) as s:
            return transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    except Exception:
        return None


def read_embeddings(paths, grid, log=print):
    """Raw int8 embeddings (64, H, W) on the grid from one or more tiles (first valid tile wins); -128 where masked."""
    out = np.full((N_AXES, grid["height"], grid["width"]), -128, "int8")
    for p in paths:
        log(f"  embeddings tile {p}")
        a = np.stack([s2fetch.warp_read(p, grid, Resampling.nearest, band=b + 1, src_nodata=-128, nodata=-128) for b in range(N_AXES)])
        empty = (out == -128).all(0)
        out[:, empty] = a[:, empty]
    if (out == -128).all():
        raise RuntimeError("no valid embedding pixels over the area (wrong year, tile or CRS?)")
    return out
