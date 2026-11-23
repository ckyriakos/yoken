"""Resolve reflectance/temperature scale + offset, in order of trust, and report the source used."""
import rasterio


def resolve_scale_offset(href, extra_fields, props, fallback=None):
    """1) STAC raster:bands  2) GeoTIFF tags  3) documented fallback (ESA rule for Sentinel-2 via
    processing baseline, or an explicit product constant). Raises if none applies."""
    rb = (extra_fields or {}).get("raster:bands")
    if rb and "scale" in rb[0]:
        return rb[0]["scale"], rb[0].get("offset", 0.0), "stac:raster:bands"
    with rasterio.open(href) as s:
        sc, of = s.scales[0], s.offsets[0]
    if (sc, of) != (1.0, 0.0):
        return sc, of, "geotiff_tags"
    pb = (props or {}).get("s2:processing_baseline")
    if pb is not None:      # ESA: baseline >= 04.00 adds a BOA offset of -1000 DN (reflectance = DN*1e-4 - 0.1)
        return 1e-4, (-0.1 if float(pb) >= 4.0 else 0.0), f"esa_rule:processing_baseline_{pb}"
    if fallback:
        return fallback[0], fallback[1], "product_constant"
    raise ValueError(f"No scale/offset found for {href}")
