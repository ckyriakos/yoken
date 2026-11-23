"""Land-cover change study: fit the response model, then evaluate edits (planting, new road) with uncertainty.

Uncertainty sources, all carried into every replicate:  (1) block-bootstrap coefficients, (2) low/base/high land-cover thresholds
(classification, siting and fit are all redone per set), (3) municipality-wide vs local fit, (4) the share of an edited cell that really
becomes canopy. What is NOT covered: air-temperature effects, indoor heat, traffic heat, tree age, irrigation, errors in the footprints."""
import numpy as np, pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from shapely.geometry import LineString, Point, Polygon, box

from . import lulc, lulc_edit as ed, response as rs
from .lulc import GROUND, TREE, GRASS, BUILDING


def _centres(transform, h, w):
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return transform.c + cols * transform.a, transform.f + rows * transform.e


class Study:
    """Everything needed to evaluate edits near one centre. Picklable (numpy, pandas, shapely only)."""

    # -- construction -------------------------------------------------------------------------------------------------------
    # cls_sets: dict name -> (hard class raster at 10 m, uint8, from lulc.classify or lulc_calib.decide; description dict).
    @classmethod
    def build(cls, L, lst, cls_sets, transform30, crs_wkt, mun_mask, pop30, streets, centre_xy, log=print):
        self = cls()
        c, m, ev = L["classes"], L["model"], L["study"]
        self.crs_wkt, self.centre, self.L = crs_wkt, tuple(centre_xy), L
        H, W = lst.shape
        X30, Y30 = _centres(transform30, H, W)
        t10 = Affine(transform30.a / 3, 0, transform30.c, 0, transform30.e / 3, transform30.f)

        # window (in 30 m pixels) around the centre
        rad = ev["zone_radius_m"] + ev.get("window_margin_m", 300)
        inv = ~transform30
        cc, rr = inv @ centre_xy
        k = int(np.ceil(rad / abs(transform30.a)))
        r0, r1, c0, c1 = max(int(rr) - k, 0), min(int(rr) + k + 1, H), max(int(cc) - k, 0), min(int(cc) + k + 1, W)
        self.win = (r0, r1, c0, c1)
        self.t30 = transform30 @ Affine.translation(c0, r0)
        self.t10 = t10 @ Affine.translation(3 * c0, 3 * r0)
        h30, w30 = r1 - r0, c1 - c0
        xw, yw = _centres(self.t30, h30, w30)
        dist = np.hypot(xw - centre_xy[0], yw - centre_xy[1])
        self.zone = dist <= ev["zone_radius_m"]
        self.lst = lst[r0:r1, c0:c1].astype("float32")
        self.pop30 = (pop30[r0:r1, c0:c1] if pop30 is not None else np.zeros((h30, w30))).astype("float32")
        self.has_pop = pop30 is not None
        x10, y10 = _centres(self.t10, 3 * h30, 3 * w30)
        self.siting_mask = np.hypot(x10 - centre_xy[0], y10 - centre_xy[1]) <= ev["siting_radius_m"]
        wbox = box(self.t10.c, self.t10.f + self.t10.e * 3 * h30, self.t10.c + self.t10.a * 3 * w30, self.t10.f)
        self.streets = streets[streets.intersects(wbox)].copy()
        hw = ev.get("street_half_width_m", 4.0)
        self.road_cov = ed.coverage([g.buffer(hw) for g in self.streets.geometry], self.t10, (3 * h30, 3 * w30))
        self.score10 = ed.score_raster(self.lst, self.pop30 if self.has_pop else None, (3 * h30, 3 * w30))

        # fit sample and model per threshold set
        trend_c = centre_xy
        xs, ys = rs.trend_coords(X30, Y30, trend_c)
        blk_b = (np.indices((H, W))[0] // m["block_px"]) * 100000 + np.indices((H, W))[1] // m["block_px"]
        blk_cv = (np.indices((H, W))[0] // m["cv_block_px"]) * 100000 + np.indices((H, W))[1] // m["cv_block_px"]
        local = np.hypot(X30 - centre_xy[0], Y30 - centre_xy[1]) <= m["local_fit_radius_m"]
        self.sets, rows_coef, rows_cv, self.problems, self.warnings = {}, [], [], [], []
        for name, (cls_, th) in cls_sets.items():
            S, valid = lulc.to_state(cls_)
            F, _ = lulc.state_fractions(S, valid)
            Ff = lulc.focal_mean(F, m["focal_px"])
            ok = mun_mask & np.isfinite(lst) & np.isfinite(Ff).all(0)
            entry = {"thresholds": th, "betas": {}, "shares": lulc.class_shares(cls_, np.kron(mun_mask, np.ones((3, 3), bool)))}
            for scope, sel in (("municipality", ok), ("local", ok & local)):
                if sel.sum() < 200:
                    log(f"  set {name}/{scope}: only {int(sel.sum())} pixels, skipped")
                    continue
                Xd = rs.design(Ff[:, sel], xs[sel], ys[sel])
                y = lst[sel].astype("float64")
                beta = rs.fit(Xd, y)
                draws = rs.block_bootstrap(Xd, y, blk_b[sel], m["n_boot"], m["seed"])
                entry["betas"][scope] = draws[:, 1:4]
                rows_coef += [{"set": name, "scope": scope, "n_pixels": int(sel.sum()), **r}
                              for r in rs.summarise_coefs(beta, draws)]
                rows_cv.append({"set": name, "scope": scope, **rs.block_cv(Xd, y, blk_cv[sel], seed=m["seed"])})
                for msg in rs.sign_problems(draws[:, 1:4]):
                    self.problems.append(f"{name}/{scope}: {msg}")
                for msg in rs.sign_warnings(draws[:, 1:4]):
                    self.warnings.append(f"{name}/{scope}: {msg}")
                if name == c["base_set"] and scope == "municipality":
                    self.hot_thr = float(np.percentile(y, 90))
            # window state
            sl10 = (slice(3 * r0, 3 * r1), slice(3 * c0, 3 * c1))
            entry["S0"], entry["valid"] = S[(slice(None),) + sl10].copy(), valid[sl10].copy()
            F0, _ = lulc.state_fractions(entry["S0"], entry["valid"])
            entry["F0f"] = lulc.focal_mean(F0, m["focal_px"])
            entry["F0"] = F0
            self.sets[name] = entry
        self.coef = pd.DataFrame(rows_coef)
        self.cv = pd.DataFrame(rows_cv)
        self.base = c["base_set"]
        return self

    # -- helpers ------------------------------------------------------------------------------------------------------------
    def to_xy(self, lonlat):
        tr = Transformer.from_crs("EPSG:4326", self.crs_wkt, always_xy=True)
        xs, ys = tr.transform([p[0] for p in lonlat], [p[1] for p in lonlat])
        return list(zip(xs, ys))

    def _plans(self, S, valid, ops):
        plans, cur = [], S
        for op in ops:
            if op["type"] == "plant" and op["mode"] == "open_land":
                p = [ed.plan_open_land(cur, valid, self.score10, self.road_cov, self.siting_mask, op["budget_ha"],
                                       op["canopy_fraction_range"], op.get("min_patch_px", 3))]
            elif op["type"] == "plant" and op["mode"] == "street":
                sub = self.streets[self.streets.distance(Point(*self.centre)) <= self.L["study"]["siting_radius_m"]]
                p = [ed.plan_street_trees(cur, valid, self.score10, self.t10, sub, op["budget_m"], op["canopy_fraction_range"],
                                          op.get("half_width_m", 4.0), allowed_highway=op.get("allowed_highway"))]
            elif op["type"] == "road":
                line = op["line_xy"] if "line_xy" in op else LineString(self.to_xy(op["coords_lonlat"]))
                p = ed.plan_road(line, op["width_m"], self.t10, cur.shape[1:], op.get("demolish_buildings", False),
                                 op.get("verge_trees_m", 0.0), tuple(op.get("verge_canopy_range", (0.3, 0.6))))
            elif op["type"] == "reclass":
                poly = op["polygon_xy"] if "polygon_xy" in op else Polygon(self.to_xy(op["polygon_lonlat"]))
                p = [ed.plan_reclass(poly, op["to"], self.t10, cur.shape[1:], op.get("from"), tuple(op.get("amount_range", (1.0, 1.0))))]
            else:
                raise ValueError(f"unknown edit {op}")
            plans += p
            cur = ed.mid_state(cur, p)
        return plans

    # -- evaluation ---------------------------------------------------------------------------------------------------------
    def evaluate(self, ops, reps_per_set=None, seed=0, maps=True):
        """Replicate-level results for a list of edits. Returns (DataFrame of replicates, info dict, maps dict)."""
        rng = np.random.default_rng(seed)
        rows, info, acc = [], {}, {"d": np.zeros(self.lst.shape), "n": 0}
        pix_ha = abs(self.t30.a * self.t30.e) / 1e4
        pop, lst, zone = self.pop30, self.lst, self.zone
        hot_thr = self.hot_thr
        for name, E in self.sets.items():
            plans = self._plans(E["S0"], E["valid"], ops)
            if name == self.base:
                info = {"plans": [{"kind": p["kind"], **p["info"]} for p in plans]}
            scopes = list(E["betas"])
            n_rep = reps_per_set or len(scopes) * len(E["betas"][scopes[0]])
            F0, F0f = E["F0"], E["F0f"]
            for r in range(n_rep):
                scope = scopes[r % len(scopes)]
                beta = E["betas"][scope][rng.integers(len(E["betas"][scope]))]
                u = rng.random(len(plans))
                S1 = E["S0"]
                for p, ui in zip(plans, u):
                    S1 = ed.apply_plan(S1, p, ui)
                F1, _ = lulc.state_fractions(S1, E["valid"])
                dFf = np.nan_to_num(lulc.focal_mean(F1, self.L["model"]["focal_px"]) - F0f)
                d = np.tensordot(beta, dFf, 1)
                ok = zone & np.isfinite(lst)
                touched = ok & (np.abs(dFf).sum(0) > 0)
                P = pop * ok
                dF = np.nan_to_num(F1 - F0)
                before_hot = (lst >= hot_thr) & ok
                after_hot = (lst + d >= hot_thr) & ok
                with np.errstate(invalid="ignore", divide="ignore"):
                    lost_b = np.where(F0[2] > 0, np.clip(-dF[2] / np.where(F0[2] > 0, F0[2], 1), 0, 1), 0)
                row = {"set": name, "scope": scope, "rep": r, "amount_u": float(u[0]) if len(u) else 0.0,
                       "zone_mean_dlst_c": float(d[ok].mean()) if ok.any() else np.nan,
                       "edited_area_mean_dlst_c": float(d[touched].mean()) if touched.any() else 0.0,
                       "popw_dlst_c": float((P * d).sum() / P.sum()) if P.sum() > 0 else np.nan,
                       "person_degc": float((P * d).sum()),
                       "pop_hot_before": float((pop * before_hot).sum()), "pop_hot_after": float((pop * after_hot).sum()),
                       "hot_ha_before": float(before_hot.sum() * pix_ha), "hot_ha_after": float(after_hot.sum() * pix_ha),
                       "pop_displaced": float((pop * lost_b * ok).sum()),
                       "ha_tree_gain": float((S1[TREE] - E["S0"][TREE]).sum() * ed.CELL_HA),
                       "ha_ground_gain": float((S1[GROUND] - E["S0"][GROUND]).sum() * ed.CELL_HA),
                       "ha_building_loss": float((E["S0"][BUILDING] - S1[BUILDING]).sum() * ed.CELL_HA),
                       "ha_grass_loss": float((E["S0"][GRASS] - S1[GRASS]).sum() * ed.CELL_HA)}
                row["d_hot_pop"] = row["pop_hot_after"] - row["pop_hot_before"]
                row["d_hot_ha"] = row["hot_ha_after"] - row["hot_ha_before"]
                rows.append(row)
                if name == self.base:
                    acc["d"] += d; acc["n"] += 1
                    last = S1
        res = pd.DataFrame(rows)
        out_maps = {}
        if maps:
            B = self.sets[self.base]
            out_maps = {"lst": self.lst, "zone": self.zone, "pop": self.pop30,
                        "d_mean": acc["d"] / max(acc["n"], 1), "class_before": B["S0"].argmax(0).astype("uint8"),
                        "class_after": last.argmax(0).astype("uint8"), "valid": B["valid"]}
        return res, info, out_maps


METRICS = ["zone_mean_dlst_c", "edited_area_mean_dlst_c", "popw_dlst_c", "person_degc", "d_hot_pop", "d_hot_ha", "pop_displaced",
           "ha_tree_gain", "ha_ground_gain", "ha_building_loss"]


def summarise(res, by, metrics=METRICS, ci=(5, 95)):
    """Mean and interval per group; also the share of replicates with a cooling sign (zone_mean_dlst_c < 0)."""
    rows = []
    for key, g in res.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_replicates"] = len(g)
        for mt in metrics:
            r[mt] = g[mt].mean()
            r[f"{mt}_p{ci[0]:02d}"], r[f"{mt}_p{ci[1]:02d}"] = np.nanpercentile(g[mt], ci)
        r["share_cooling_pct"] = 100 * float((g.zone_mean_dlst_c < 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def vkey(v):
    """Canonical text for a sweep value (24, 24.0 -> '24'; 0.25 -> '0.25'; 'tree' -> 'tree')."""
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v)
