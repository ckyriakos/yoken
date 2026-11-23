"""Land-cover edits on the 10 m state array S (4, h, w) = shares of ground, tree, grass, building per cell.

An edit is a `plan`: a coverage array (0..1 per cell), the classes it takes area from, the class it gives area to, and an
amount range (share of the covered cell that changes; drawn once per replicate). `apply_plan` moves area and conserves each cell.
Siting (where to plant) is deterministic given the state and the score; only the amount is random."""
import math
import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.transform import Affine
from scipy.ndimage import label
from shapely.geometry import LineString
from shapely.ops import substring

from .lulc import GROUND, TREE, GRASS, BUILDING

CELL_HA = 100.0 / 10000.0                            # one 10 m cell in hectares


def coverage(geoms, transform, shape, ss=5):
    """Share (0..1) of each cell covered by the geometries, rasterised at cell/ss."""
    h, w = shape
    shapes = [(g, 1) for g in geoms if g is not None and not g.is_empty]
    if not shapes:
        return np.zeros(shape, "float32")
    a = rasterize(shapes, out_shape=(h * ss, w * ss), transform=transform @ Affine.scale(1 / ss), fill=0, dtype="uint8")
    return a.reshape(h, ss, w, ss).mean(axis=(1, 3)).astype("float32")


def reclass(S, cov, to, from_ids, amount):
    """Move amount*cov of each cell's area from `from_ids` (proportionally to what is there) to `to`."""
    if to in from_ids:
        raise ValueError("target class cannot be a source class")
    S = S.copy()
    ids = list(from_ids)
    avail = S[ids].sum(0)
    move = np.minimum(amount * cov, avail)
    take = move / np.where(avail > 0, avail, 1.0)
    for k in ids:
        S[k] -= S[k] * take
    S[to] += move
    return S


def apply_plan(S, plan, u):
    """u in [0,1] picks the amount inside the plan's range."""
    lo, hi = plan["amount_range"]
    return reclass(S, plan["cov"], plan["to"], plan["from"], lo + u * (hi - lo))


def mid_state(S, plans):
    for p in plans:
        S = apply_plan(S, p, 0.5)
    return S


# --- siting ---------------------------------------------------------------------------------------------------------------
def score_raster(heat30, pop30, shape10):
    """Where planting matters most: people x heat. Both min-max scaled inside the window, 30 m -> 10 m by repetition.
    No population -> heat only."""
    def mm(a):
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 0 else np.ones_like(a)
    s = mm(heat30) * (mm(pop30) if pop30 is not None and np.nanmax(pop30) > 0 else 1.0)
    return np.kron(s, np.ones((3, 3)))[:shape10[0], :shape10[1]].astype("float32")


def plan_open_land(S, valid, score10, road_cov, allowed, budget_ha, amount_range, min_patch_px=3, ground_min=0.8, road_max=0.2):
    """Plant on open ground that is not street, not building and not already green (plazas, lots, bare plots).
    Whole connected patches are taken in order of mean score; the last patch is trimmed to the budget."""
    cand = valid & allowed & (S[GROUND] >= ground_min) & (road_cov < road_max)
    lab, n = label(cand, structure=np.ones((3, 3)))
    cov = np.zeros(S.shape[1:], "float32")
    budget_px = int(round(budget_ha / CELL_HA))
    capacity_ha = float(cand.sum() * CELL_HA)
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        means = np.bincount(lab.ravel(), weights=score10.ravel(), minlength=n + 1) / np.maximum(sizes, 1)
        left = budget_px
        for k in np.argsort(-means[1:]) + 1:
            if left <= 0:
                break
            if sizes[k] < min_patch_px:
                continue
            m = lab == k
            if sizes[k] <= left:
                cov[m] = 1.0
                left -= int(sizes[k])
            elif left >= min_patch_px:
                idx = np.flatnonzero(m.ravel())
                top = idx[np.argsort(-score10.ravel()[idx])[:left]]
                cov.ravel()[top] = 1.0
                left = 0
    chosen_ha = float(cov.sum() * CELL_HA)
    return {"kind": "plant_open_land", "cov": cov, "to": TREE, "from": (GROUND,), "amount_range": tuple(amount_range),
            "info": {"selected_ha": round(chosen_ha, 3), "capacity_ha": round(capacity_ha, 3),
                     "budget_ha": budget_ha, "budget_unmet_ha": round(max(budget_ha - chosen_ha, 0.0), 3)}}


def _pieces(line, piece_m):
    n = max(1, math.ceil(line.length / piece_m))
    return [substring(line, line.length * i / n, line.length * (i + 1) / n) for i in range(n)]


def plan_street_trees(S, valid, score10, transform, streets, budget_m, amount_range, half_width_m=4.0, piece_m=50.0,
                      max_existing_tree=0.3, allowed_highway=None):
    """Canopy over street pieces (kerb trees overhanging the carriageway): best-scoring 50 m pieces until the length budget is
    used. `streets` is a GeoDataFrame in the raster CRS with a `highway` column. Takes area from ground only."""
    h, w = S.shape[1:]
    ss = streets if allowed_highway is None else streets[streets.highway.isin(allowed_highway)]
    geoms = []
    for g in ss.geometry:
        for part in (g.geoms if hasattr(g, "geoms") else [g]):
            if isinstance(part, LineString) and part.length > 1:
                geoms.extend(_pieces(part, piece_m))
    inv = ~transform
    rows_ = []
    for g in geoms:
        pts = [g.interpolate(f, normalized=True) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        cr = [inv @ (p.x, p.y) for p in pts]
        ij = [(math.floor(r), math.floor(c)) for c, r in cr]
        ij = [(r, c) for r, c in ij if 0 <= r < h and 0 <= c < w]
        if not ij:
            continue
        sc = float(np.mean([score10[r, c] for r, c in ij]))
        tr = float(np.mean([S[TREE, r, c] for r, c in ij]))
        gr = float(np.mean([S[GROUND, r, c] for r, c in ij]))
        if tr <= max_existing_tree and gr >= 0.3:
            rows_.append((sc, g))
    rows_.sort(key=lambda t: -t[0])
    chosen, tot = [], 0.0
    for sc, g in rows_:
        if tot >= budget_m:
            break
        chosen.append(g); tot += g.length
    cov = coverage([g.buffer(half_width_m, cap_style="flat") for g in chosen], transform, (h, w))
    return {"kind": "plant_street_trees", "cov": cov, "to": TREE, "from": (GROUND,), "amount_range": tuple(amount_range),
            "info": {"selected_m": round(tot, 1), "budget_m": budget_m, "candidate_m": round(sum(g.length for _, g in rows_), 1),
                     "budget_unmet_m": round(max(budget_m - tot, 0.0), 1)}}


def plan_road(line, width_m, transform, shape, demolish_buildings=False, verge_trees_m=0.0, verge_amount_range=(0.3, 0.6)):
    """New carriageway (width_m, flat ends). It becomes ground (asphalt) wherever it crosses trees, grass or, if
    demolish_buildings, buildings. Optional tree verge of verge_trees_m on each side (canopy share drawn from the range).
    Returns a list of plans (road first, verge second)."""
    road = line.buffer(width_m / 2, cap_style="flat")
    cov_road = coverage([road], transform, shape)
    from_ids = (TREE, GRASS, BUILDING) if demolish_buildings else (TREE, GRASS)
    plans = [{"kind": "road", "cov": cov_road, "to": GROUND, "from": from_ids, "amount_range": (1.0, 1.0),
              "info": {"width_m": width_m, "length_m": round(line.length, 1), "area_ha": round(road.area / 1e4, 3),
                       "demolish_buildings": bool(demolish_buildings)}}]
    if verge_trees_m > 0:
        band = line.buffer(width_m / 2 + verge_trees_m, cap_style="flat").difference(road)
        plans.append({"kind": "road_verge_trees", "cov": coverage([band], transform, shape), "to": TREE, "from": (GROUND,),
                      "amount_range": tuple(verge_amount_range),
                      "info": {"verge_m": verge_trees_m, "area_ha": round(band.area / 1e4, 3)}})
    return plans


CLASS_ID = {"ground": GROUND, "tree": TREE, "grass": GRASS, "building": BUILDING}


def plan_reclass(polygon, to, transform, shape, from_classes=None, amount_range=(1.0, 1.0)):
    """General land-cover change: everything inside the polygon (or `from_classes` only) becomes class `to`
    (ground, tree, grass or building); the share of each cell that changes is drawn from amount_range."""
    t = CLASS_ID[to]
    src = tuple(CLASS_ID[c] for c in from_classes) if from_classes else tuple(k for k in CLASS_ID.values() if k != t)
    return {"kind": "reclass", "cov": coverage([polygon], transform, shape), "to": t, "from": src, "amount_range": tuple(amount_range),
            "info": {"to": to, "area_ha": round(polygon.area / 1e4, 3)}}
