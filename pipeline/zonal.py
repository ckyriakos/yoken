"""Assign raster pixels to neighbourhoods (cell-centre rule) and collect per-polygon pixel arrays."""
import numpy as np
from rasterio.features import rasterize


def label_grid(gdf, profile):
    """Integer label raster: row position of each polygon in gdf (0..n-1), -1 outside."""
    g = gdf.to_crs(profile["crs"])
    shapes = [(geom, i) for i, geom in enumerate(g.geometry)]
    return rasterize(shapes, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=-1, dtype="int32", all_touched=False)


def pixel_table(labels, layers, block_px, n):
    """Per neighbourhood: dict(layer arrays, 'blk' block ids). Pixels keep NaNs so coverage is computable."""
    rows, cols = np.indices(labels.shape)
    blk = (rows // block_px) * 100000 + (cols // block_px)
    out = {}
    for i in range(n):
        m = labels == i
        out[i] = {"blk": blk[m], **{k: v[m] for k, v in layers.items()}}
    return out
