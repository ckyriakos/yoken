"""Building-shadow model for street edges. Shade from BUILDINGS ONLY (no trees, awnings or arcades), so shade is underestimated.
Sun position: NOAA-style approximation (about 0.5 degree). Shadow of a building = convex hull of its footprint and the footprint shifted
by height/tan(altitude) away from the sun; this over-covers courtyards and concave footprints slightly."""
import math
from datetime import datetime, timezone

import numpy as np
import shapely
from shapely import STRtree


def sun_position(lat, lon, when_utc):
    """Return (altitude_deg, azimuth_deg clockwise from north) for a timezone-aware UTC datetime."""
    d = when_utc.astimezone(timezone.utc)
    n = d.timetuple().tm_yday
    frac_year = 2 * math.pi / 365 * (n - 1 + (d.hour - 12) / 24 + d.minute / 1440)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(frac_year) - 0.032077 * math.sin(frac_year)
                    - 0.014615 * math.cos(2 * frac_year) - 0.040849 * math.sin(2 * frac_year))
    decl = (0.006918 - 0.399912 * math.cos(frac_year) + 0.070257 * math.sin(frac_year) - 0.006758 * math.cos(2 * frac_year)
            + 0.000907 * math.sin(2 * frac_year) - 0.002697 * math.cos(3 * frac_year) + 0.00148 * math.sin(3 * frac_year))
    tst = d.hour * 60 + d.minute + eqt + 4 * lon                        # true solar time, minutes
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cosz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    zen = math.acos(max(-1.0, min(1.0, cosz)))
    alt = 90 - math.degrees(zen)
    az = math.degrees(math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(decl) * math.cos(la))) + 180
    return alt, az % 360


def shadows(footprints, heights, alt_deg, az_deg, min_alt=10.0):
    """footprints: array of shapely polygons (metric CRS), heights in m. Returns array of shadow polygons (footprint included).
    Altitude below min_alt is clamped so shadow length stays finite."""
    alt = max(alt_deg, min_alt)
    L = np.asarray(heights, float) / math.tan(math.radians(alt))
    az = math.radians(az_deg)
    off = np.c_[-L * math.sin(az), -L * math.cos(az)]                    # shadow falls away from the sun
    g = np.asarray(footprints, dtype=object)
    xy, idx = shapely.get_coordinates(g, return_index=True)
    moved = shapely.set_coordinates(g.copy(), xy + off[idx])
    return shapely.convex_hull(shapely.union(g, moved))


def sample_edges(edges, step_m):
    """edges: list of LineStrings. Returns (points array, edge index per point). Every edge gets at least its midpoint."""
    pts, owner = [], []
    for i, e in enumerate(edges):
        n = max(1, int(round(e.length / step_m)))
        for k in range(n):
            pts.append(e.interpolate((k + 0.5) / n, normalized=True)); owner.append(i)
    return np.array(pts, dtype=object), np.array(owner)


def shade_fraction(points, owner, n_edges, shadow_polys):
    """Share of sample points per edge that lie inside any shadow polygon."""
    tree = STRtree(shadow_polys)
    hit = np.unique(tree.query(points, predicate="within")[0])
    shaded = np.zeros(len(points), bool)
    shaded[hit] = True
    cnt = np.bincount(owner, minlength=n_edges)
    return np.bincount(owner, weights=shaded, minlength=n_edges) / np.maximum(cnt, 1)
