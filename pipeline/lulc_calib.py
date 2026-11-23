"""Calibrated land cover: a classifier fitted to reference labels instead of hand-set NDVI cut-offs.

Classes the classifier decides: ground (paved or bare), tree, grass. 'Building' always comes from the footprints (a cell with at least
building_min_cover of footprint), because roofs and roads look alike to Sentinel-2. Features are any mix of spectral indices and
embedding axes. Reference labels are ESA WorldCover and/or the user's own labelled points or polygons.

Honest limits: WorldCover is itself a model (about 75-85% accurate globally, dated 2020/2021), so 'accuracy' means agreement with the
reference, not with the ground. Spatial block cross-validation keeps nearby pixels out of the same fold, so the numbers are not inflated by
neighbouring pixels. Probabilities are re-weighted from the balanced training sample to the true class frequencies before decisions."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .lulc import GROUND, TREE, GRASS, BUILDING, NODATA, CLASS_NAMES

WORLDCOVER_DEFAULT = {10: TREE, 20: GRASS, 30: GRASS, 40: GRASS, 50: GROUND, 60: GROUND}     # others (water, wetland, snow...) excluded
CLASSIFIED = (GROUND, TREE, GRASS)


class FeatureStack:
    """Feature access without materialising a (pixels x features) matrix for the whole grid."""

    def __init__(self, indices=None, emb_raw=None):
        self.idx = indices or {}
        self.emb = emb_raw
        self.names = list(self.idx) + ([f"A{i:02d}" for i in range(emb_raw.shape[0])] if emb_raw is not None else [])
        ref = next(iter(self.idx.values())) if self.idx else emb_raw[0]
        self.shape = ref.shape[-2:]

    def take(self, flat):
        cols = [v.ravel()[flat] for v in self.idx.values()]
        if self.emb is not None:
            from .embeddings import dequantize
            e = self.emb.reshape(self.emb.shape[0], -1)[:, flat]
            cols += list(dequantize(e))
        return np.column_stack(cols).astype("float32")

    def valid(self):
        ok = np.ones(self.shape, bool)
        for v in self.idx.values():
            ok &= np.isfinite(v)
        if self.emb is not None:
            ok &= (self.emb != -128).all(0)
        return ok


def worldcover_to_classes(wc, mapping=None):
    m = mapping or WORLDCOVER_DEFAULT
    out = np.full(wc.shape, NODATA, "uint8")
    for code, cls in m.items():
        out[wc == int(code)] = cls
    return out


def labels_from_geometries(gdf, class_col, grid):
    """Rasterise user labels (points or polygons with a text class column) onto the grid; unlabeled = NODATA."""
    from rasterio.features import rasterize
    ids = {n: i for i, n in enumerate(CLASS_NAMES)}
    bad = sorted(set(gdf[class_col].astype(str)) - set(ids))
    if bad:
        raise ValueError(f"label classes must be one of {list(ids)}, found {bad}")
    g = gdf.to_crs(grid["crs"])
    r = 0.75 * abs(grid["transform"].a)                      # a point labels the pixel(s) whose centre lies within 0.75 pixel
    shapes = [(geom.buffer(r) if geom.geom_type == "Point" else geom, ids[str(c)]) for geom, c in zip(g.geometry, g[class_col])]
    return rasterize(shapes, out_shape=(grid["height"], grid["width"]), transform=grid["transform"], fill=NODATA, dtype="uint8")


def merge_labels(*layers):
    """Later layers override earlier ones wherever they have a label."""
    out = layers[0].copy()
    for l in layers[1:]:
        out = np.where(l != NODATA, l, out)
    return out


def sample_pixels(ref, usable, n_per_class, rng):
    """Stratified sample of flat indices per classified class (fewer if a class is rare)."""
    picks = []
    for c in CLASSIFIED:
        idx = np.flatnonzero(((ref == c) & usable).ravel())
        if len(idx):
            picks.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    return np.concatenate(picks)


def _scores(y, p):
    cm = pd.crosstab(pd.Series(y, name="reference"), pd.Series(p, name="predicted")).reindex(index=CLASSIFIED, columns=CLASSIFIED, fill_value=0)
    n = cm.values.sum()
    oa = float(np.trace(cm.values) / max(n, 1))
    pe = float((cm.values.sum(0) * cm.values.sum(1)).sum() / max(n * n, 1))
    rows = []
    for k, c in enumerate(CLASSIFIED):
        tp, fp, fn = cm.values[k, k], cm.values[:, k].sum() - cm.values[k, k], cm.values[k, :].sum() - cm.values[k, k]
        rows.append({"class": CLASS_NAMES[c], "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)), "n_reference": int(cm.values[k].sum())})
    return {"overall_accuracy": oa, "kappa": float((oa - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "confusion_matrix": {"rows_reference": [CLASS_NAMES[c] for c in CLASSIFIED], "cols_predicted": [CLASS_NAMES[c] for c in CLASSIFIED],
                                 "values": cm.values.tolist()}, "per_class": rows}


def calibrate(feats, ref, train_ok, pred_ok, grid_res_m, cfg, ndvi=None, rule_thresholds=None, seed=0, log=print):
    """Fit, cross-validate and predict class probabilities for every usable pixel.
    feats: FeatureStack. ref: uint8 labels (NODATA where unknown/excluded). train_ok: bool, where samples and class frequencies may come
    from (the municipality, no buildings). pred_ok: bool, where probabilities are produced (also the surroundings, no buildings).
    Returns (P (3,H,W) float32 prior-corrected, NaN where not predicted; qc dict)."""
    rng = np.random.default_rng(seed)
    idx = sample_pixels(ref, train_ok & feats.valid(), cfg.get("n_per_class", 4000), rng)
    if len(idx) < 300 or len(set(ref.ravel()[idx])) < 3:
        raise RuntimeError(f"too few labelled pixels per class to calibrate ({len(idx)} sampled, classes {sorted(set(ref.ravel()[idx]))})")
    X, y = feats.take(idx), ref.ravel()[idx].astype(int)
    H, W = feats.shape
    r, c = np.divmod(idx, W)
    bpx = max(int(cfg.get("cv_block_m", 600) / grid_res_m), 1)
    groups = (r // bpx) * 100000 + (c // bpx)
    make = lambda: HistGradientBoostingClassifier(max_iter=cfg.get("max_iter", 150), learning_rate=0.1, max_leaf_nodes=15,
                                                  class_weight="balanced", random_state=seed)
    oof = np.empty_like(y)
    k = min(5, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
    qc = {"n_train": int(len(idx)), "n_blocks": int(len(np.unique(groups))), "cv_folds": int(k), "features": feats.names,
          "cv": _scores(y, oof)}
    if ndvi is not None and rule_thresholds:
        v = ndvi.ravel()[idx]
        rp = np.where(v >= rule_thresholds["tree"], TREE, np.where(v >= rule_thresholds["grass"], GRASS, GROUND))
        qc["rule_baseline_cv"] = {"thresholds": rule_thresholds, **{k_: _scores(y, rp)[k_] for k_ in ("overall_accuracy", "kappa")}}
    model = make().fit(X, y)
    # prior correction: training was balanced, reality is not
    true_freq = np.array([((ref == c) & train_ok).sum() for c in CLASSIFIED], float)
    true_freq /= true_freq.sum()
    qc["reference_class_share_pct"] = {CLASS_NAMES[c]: round(100 * float(f), 2) for c, f in zip(CLASSIFIED, true_freq)}
    w = true_freq / (1 / 3)
    order = [list(model.classes_).index(c) for c in CLASSIFIED]
    P = np.full((3, H * W), np.nan, "float32")
    ok = np.flatnonzero((feats.valid() & pred_ok).ravel())
    for s in range(0, len(ok), 200_000):
        chunk = ok[s:s + 200_000]
        p = model.predict_proba(feats.take(chunk))[:, order] * w
        P[:, chunk] = (p / p.sum(1, keepdims=True)).T
    return P.reshape(3, H, W), qc


def decide(P, bld, valid_any, tau_tree, tau_grass, bld_min):
    """Hard classes from probabilities: tree if P(tree) >= tau_tree, else grass if P(grass) >= tau_grass, else ground.
    Footprint cells become building; cells without features and without a building are NODATA."""
    cls = np.full(P.shape[1:], GROUND, "uint8")
    cls[np.nan_to_num(P[2]) >= tau_grass] = GRASS
    cls[np.nan_to_num(P[1]) >= tau_tree] = TREE
    cls[~valid_any] = NODATA
    cls[bld >= bld_min] = BUILDING
    return cls
