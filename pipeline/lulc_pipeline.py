"""Orchestration for the land-cover what-if, shared by the command-line scripts and the app.

build_sets(cfg, L, out)       data -> land-cover class sets (selectable source, indices, classifier) -> files in `out`
make_study(cfg, L, out)       class sets + LST + population + streets -> fitted Study
run_scenarios(study, L, out)  every scenario in lulc.yaml -> summary, runs, maps, geometry, pickle, assumptions
The `data:` block of lulc.yaml selects the source (local files or live STAC), the spectral indices, the features (indices, embeddings
or both) and the classifier (hand-set rules or calibrated against reference labels)."""
import copy, hashlib, json, pickle
from pathlib import Path

import geopandas as gpd, numpy as np, pandas as pd, rasterio
import rasterio.crs, rasterio.transform
from pyproj import Transformer
from rasterio.enums import Resampling
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import linemerge, unary_union

from . import buildings, embeddings, indices as ix, landuse, lulc, lulc_calib as cal, lulc_run as lr, rasters, s2fetch, zonal

DATA_DEFAULTS = {"source": "local", "indices": ["ndvi"], "features": "indices",
                 "live": {"year": 2026, "months": [6, 7, 8, 9], "max_cloud_pct": 20, "max_scenes": 8, "min_valid_scenes": 3},
                 "embeddings": {"year": 2025, "paths": []},
                 "classifier": {"method": "rules", "calibrated": {"reference": ["worldcover"], "worldcover_year": 2021, "n_per_class": 4000,
                                                                   "cv_block_m": 600, "tau": {"low": 0.35, "base": 0.5, "high": 0.65}}}}


def data_cfg(L):
    """lulc.yaml data block with defaults filled in (nested dicts merged)."""
    def merge(a, b):
        o = copy.deepcopy(a)
        for k, v in (b or {}).items():
            o[k] = merge(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else copy.deepcopy(v)
        return o
    return merge(DATA_DEFAULTS, L.get("data"))


def data_hash(L):
    return hashlib.sha1(json.dumps(data_cfg(L), sort_keys=True, default=str).encode()).hexdigest()[:8]


def rule_sets(ndvi10, bld10, L):
    c = L["classes"]
    return {name: (lulc.classify(ndvi10, bld10, th["tree"], th["grass"], c["building_min_cover"]), dict(th)) for name, th in c["sets"].items()}


def _local_ndvi(cfg, grid, log):
    raw = Path(cfg["paths"]["raw"])
    scenes = [s for s in cfg["scenes"] if s["role"] in cfg["composite"]["roles_used"]]
    stack = []
    for s in scenes:
        sm = json.load(open(raw / "s2" / f"{s['s2']}_meta.json"))
        nd, _, _ = rasters.ndvi_on_grid(raw / "s2" / f"{s['s2']}_B04.tif", raw / "s2" / f"{s['s2']}_B08.tif", raw / "s2" / f"{s['s2']}_SCL.tif",
                                        grid, sm["red_scale"], sm["red_offset"], sm["nir_scale"], sm["nir_offset"])
        stack.append(nd)
        log(f"  {s['s2']}: valid {100 * np.isfinite(nd).mean():.1f}%")
    ndvi, _ = rasters.composite(np.stack(stack), cfg["composite"]["min_valid_scenes"])
    return ndvi, [{"id": s["s2"]} for s in scenes]


def build_sets(cfg, L, out, log=print, cache_dir="data/cache", stac_client=None):
    """Write the land-cover products for the selected data and classifier into `out`; returns the quality dict."""
    D = data_cfg(L)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    g, ctx = cfg["gis"], cfg.get("context", {})
    method, features, source = D["classifier"]["method"], D["features"], D["source"]
    if method == "rules" and features != "indices":
        raise ValueError("embeddings need classifier.method: calibrated (hand-set rules only work on spectral indices)")
    names = list(D["indices"])
    if method == "rules" and "ndvi" not in names:
        names = ["ndvi"] + names
    if source == "local" and set(names) != {"ndvi"}:
        raise ValueError(f"source: local only has red and near-infrared, so only the ndvi index is possible (asked for {names}); set data.source: live")

    with rasterio.open(Path(cfg["paths"]["out"]) / "lst_composite.tif") as src:
        prof = src.profile
    grid = lulc.grid10(prof)
    bbox = s2fetch.grid_bbox_wgs84(grid)
    log(f"10 m grid {grid['height']} x {grid['width']} in {prof['crs']}; source {source}, indices {names}, features {features}, classifier {method}")
    qc = {"data": D}

    # --- spectral indices
    idx, ndvi_ref = {}, None
    if features in ("indices", "both") or method == "rules":
        if source == "live":
            items = s2fetch.search_scenes(bbox, D["live"], stac_client)
            if len(items) < D["live"]["min_valid_scenes"]:
                raise RuntimeError(f"only {len(items)} scenes found; relax max_cloud_pct, months or dates in data.live")
            log(f"Live Sentinel-2: {len(items)} scenes from tile {items[0].properties.get('s2:mgrs_tile')}")
            idx, nvalid = s2fetch.composite_indices(items, names, grid, D["live"]["min_valid_scenes"], cache_dir, log)
            qc["scenes"] = s2fetch.scene_table(items)
        else:
            nd, qc["scenes"] = _local_ndvi(cfg, grid, log)
            idx = {"ndvi": nd}
    # --- embeddings
    emb = None
    if features in ("embeddings", "both"):
        E = D["embeddings"]
        ck = Path(cache_dir) / "emb" / f"{s2fetch.grid_key(grid)}_{E['year']}.npy"
        if ck.exists():
            emb = np.load(ck)
        else:
            paths = list(E.get("paths") or []) or embeddings.discover_tiles(bbox, E["year"])
            if not paths:
                raise RuntimeError(f"no embedding tiles found for {E['year']}; set data.embeddings.paths to local or remote files")
            emb = embeddings.read_embeddings(paths, grid, log)
            ck.parent.mkdir(parents=True, exist_ok=True)
            np.save(ck, emb)
        qc["embeddings"] = {"year": E["year"], "axes": embeddings.N_AXES, "share_valid_pct": round(100 * float((emb != -128).all(0).mean()), 1)}

    # --- footprints, municipality
    b, qc["building_data"] = buildings.clean(gpd.read_file(ctx.get("buildings_path", "data/raw/gis/solar_efficiency.geojson")).to_crs(2100))
    bld = lulc.building_cover(b.to_crs(prof["crs"]).geometry.to_numpy(), grid)
    nb = gpd.read_file(g["neighbourhoods_path"])
    mun30 = zonal.label_grid(nb, prof) >= 0
    mun10 = np.kron(mun30, np.ones((3, 3), bool))
    bmin = L["classes"]["building_min_cover"]
    isb = bld >= bmin

    # --- classes
    base = L["classes"]["base_set"]
    if method == "rules":
        sets = rule_sets(idx["ndvi"], bld, L)
    else:
        C = D["classifier"]["calibrated"]
        layers = []
        if "worldcover" in C["reference"]:
            if C.get("worldcover_path"):
                wc = s2fetch.warp_read(C["worldcover_path"], grid, Resampling.nearest)
            else:
                wc = s2fetch.fetch_worldcover(bbox, grid, C["worldcover_year"], stac_client, log)
            layers.append(cal.worldcover_to_classes(wc, C.get("worldcover_map")))
        if "labels" in C["reference"]:
            layers.append(cal.labels_from_geometries(gpd.read_file(C["labels_path"]), C.get("labels_class_col", "class"), grid))
        if not layers:
            raise ValueError("classifier.calibrated.reference must include worldcover and/or labels")
        ref = cal.merge_labels(*layers)
        ref[isb | (ref == lulc.BUILDING)] = lulc.NODATA
        feats = cal.FeatureStack(idx if features in ("indices", "both") else None, emb)
        th_base = L["classes"]["sets"][base]
        P, qc["calibration"] = cal.calibrate(feats, ref, mun10 & ~isb, ~isb, 10.0, C, ndvi=idx.get("ndvi"),
                                             rule_thresholds=th_base if "ndvi" in idx else None, seed=L["model"]["seed"], log=log)
        valid_any = feats.valid()
        sets = {n: (cal.decide(P, bld, valid_any, t, t, bmin), {"tau_tree": t, "tau_grass": t}) for n, t in C["tau"].items()}
        qc["calibration"]["decision_thresholds"] = C["tau"]
    if base not in sets:
        raise ValueError(f"base_set '{base}' not among class sets {list(sets)}")
    cls = sets[base][0]

    # --- write products
    p10 = {"driver": "GTiff", "height": grid["height"], "width": grid["width"], "count": 1, "crs": grid["crs"], "transform": grid["transform"],
           "compress": "deflate"}
    outputs = {"lulc_bld10": (bld, "float32", None), "lulc_10m": (cls, "uint8", 255)}
    if "ndvi" in idx:
        outputs["lulc_ndvi10"] = (idx["ndvi"], "float32", np.nan)
    for name, (arr, dt, nod) in outputs.items():
        with rasterio.open(out / f"{name}.tif", "w", dtype=dt, nodata=nod, **p10) as d:
            d.write(arr.astype(dt), 1)
    np.savez_compressed(out / "lulc_sets.npz", **{n: c_ for n, (c_, _) in sets.items()})
    if idx:
        np.savez_compressed(out / "lulc_indices.npz", **idx)
    qc["set_descriptions"] = {n: d for n, (_, d) in sets.items()}

    # --- quality
    qc["thresholds_base"] = sets[base][1]
    qc["building_min_cover"] = bmin
    qc["class_share_pct_municipality"] = lulc.class_shares(cls, mun10)
    qc["nodata_share_pct_municipality"] = round(100 * float((cls[mun10] == lulc.NODATA).mean()), 2)
    qc["index_by_class_median"] = {n: {nm: (round(float(np.nanmedian(v[mun10 & (cls == k)])), 3) if (mun10 & (cls == k)).any() else None)
                                       for k, nm in enumerate(lulc.CLASS_NAMES)} for n, v in idx.items()}
    if "ndvi" in idx:
        nd = idx["ndvi"]
        qc["ndvi_percentiles_municipality"] = {str(k): round(float(np.nanpercentile(nd[mun10], k)), 3) for k in (5, 25, 50, 75, 90, 95)}
        qc["ndvi_by_class_median"] = qc["index_by_class_median"]["ndvi"]
        h30, w30 = grid["height"] // 3, grid["width"] // 3
        nd30 = np.nanmean(nd[:3 * h30, :3 * w30].reshape(h30, 3, w30, 3), axis=(1, 3))
        ref_p = Path(cfg["paths"]["out"]) / "ndvi_composite.tif"
        if ref_p.exists():
            with rasterio.open(ref_p) as s:
                r30 = s.read(1)
            ok = mun30 & np.isfinite(nd30) & np.isfinite(r30)
            qc["agreement_with_ndvi_composite_30m"] = {"pearson_r": round(float(np.corrcoef(nd30[ok], r30[ok])[0, 1]), 3),
                                                       "median_difference": round(float(np.median(nd30[ok] - r30[ok])), 4), "n_pixels": int(ok.sum())}
    lu_path = Path(ctx.get("landuse_path", "data/raw/gis/landuse.geojson"))
    if lu_path.exists():
        from rasterio.features import rasterize
        import yaml
        lu = gpd.read_file(lu_path).to_crs(2100)
        lu["geometry"] = lu.geometry.make_valid()
        lu["landuse_class"] = landuse.classify_landuse(lu, yaml.safe_load(open("landuse_classes.yaml"))["rules"])
        nm_ = sorted(lu.landuse_class.unique())
        lu = lu.to_crs(prof["crs"])
        z = rasterize(((gm, nm_.index(k) + 1) for gm, k in zip(lu.geometry, lu.landuse_class) if gm is not None and not gm.is_empty),
                      out_shape=(grid["height"], grid["width"]), transform=grid["transform"], fill=0, dtype="uint8")
        rows = [{"zoning_class": nm, "ha": round(float((mun10 & (z == i)).sum() / 100), 1), **lulc.class_shares(cls, mun10 & (z == i))}
                for i, nm in enumerate(nm_, 1) if (mun10 & (z == i)).any()]
        zt = pd.DataFrame(rows)
        zt.to_csv(out / "lulc_vs_zoning.csv", index=False)
        gr = zt[zt.zoning_class == "green_open"]
        if len(gr):
            qc["green_zoned_share_tree_or_grass_pct"] = round(float(gr.tree.iloc[0] + gr.grass.iloc[0]), 1)
    json.dump(qc, open(out / "lulc_quality.json", "w"), indent=2, default=str)
    return qc


def make_study(cfg, L, out, log=print):
    """Fit the response model for the class sets in `out` (written by build_sets)."""
    out, gcfg, ac = Path(out), cfg["gis"], cfg.get("access", {})
    rout = Path(cfg["paths"]["out"])                                  # LST composite and population live in the main outputs
    with rasterio.open(rout / "lst_composite.tif") as s:
        lst, prof = s.read(1), s.profile
    crs = prof["crs"]
    H, W = lst.shape
    nb = gpd.read_file(gcfg["neighbourhoods_path"])
    mun = zonal.label_grid(nb, prof) >= 0
    pop30 = None
    bpf = rout / "building_population.csv"
    if bpf.exists():
        bp = pd.read_csv(bpf)
        col = "pop_cal" if "pop_cal" in bp else "pop_base"
        pts = gpd.GeoSeries(gpd.points_from_xy(bp.x, bp.y), crs=2100).to_crs(crs)
        r, c = rasterio.transform.rowcol(prof["transform"], pts.x.to_numpy(), pts.y.to_numpy())
        r, c = np.asarray(r), np.asarray(c)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        pop30 = np.zeros((H, W))
        np.add.at(pop30, (r[ok], c[ok]), bp[col].to_numpy()[ok])
        log(f"Population column {col}: {pop30.sum():,.0f} people on the grid ({int((~ok).sum())} buildings outside the raster)")
    else:
        log("building_population.csv not found: results are area-based only (no population weighting)")
    streets = gpd.read_file(ac.get("streets_path", "data/raw/gis/Streets_Center.geojson")).to_crs(crs)
    streets["highway"] = streets["highway"].fillna("unknown") if "highway" in streets else "unknown"
    streets = streets[["highway", "geometry"]]
    centre = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(*L["study"]["centre_lonlat"])
    log(f"Centre (Kallimarmaro) in raster CRS: {centre[0]:.0f}, {centre[1]:.0f}")
    z = np.load(out / "lulc_sets.npz")
    desc = json.load(open(out / "lulc_quality.json")).get("set_descriptions", {})
    sets = {n: (z[n], desc.get(n, {})) for n in z.files}
    return lr.Study.build(L, lst, sets, prof["transform"], crs.to_wkt(), mun, pop30, streets, centre, log=log)


def run_scenarios(study, L, out, reps=None, log=print):
    out = Path(out)
    crs = rasterio.crs.CRS.from_wkt(study.crs_wkt)
    centre = study.centre
    wbox = box(study.t10.c, study.t10.f + study.t10.e * 3 * study.lst.shape[0], study.t10.c + study.t10.a * 3 * study.lst.shape[1], study.t10.f)

    def resolve_alignment(name):
        a = L["alignments"][name]
        if a.get("geojson_path"):
            m = linemerge(unary_union(list(gpd.read_file(a["geojson_path"]).to_crs(crs).geometry)))
            if m.geom_type != "LineString":
                raise ValueError(f"alignment {name}: {a['geojson_path']} must contain one connected LineString")
            return m
        return LineString(study.to_xy(a["coords_lonlat"]))

    def resolve_polygon(name):
        a = L["polygons"][name]
        if a.get("geojson_path"):
            return unary_union(list(gpd.read_file(a["geojson_path"]).to_crs(crs).geometry))
        return Polygon(study.to_xy(a["coords_lonlat"]))

    lines = {}
    for nm in {op["alignment"] for s in L["scenarios"].values() for op in s["ops"] if "alignment" in op}:
        ln = resolve_alignment(nm)
        inside = ln.intersection(wbox).length / ln.length
        if inside < 0.99:
            raise ValueError(f"alignment {nm} lies {100 * (1 - inside):.0f}% outside the study window (zone radius + margin around the centre)")
        if max(Point(p).distance(Point(*centre)) for p in ln.coords) > L["study"]["siting_radius_m"] + 300:
            log(f"WARNING alignment {nm}: reaches beyond the siting radius; results beyond {L['study']['zone_radius_m']} m are not counted")
        lines[nm] = ln
    polys = {nm: resolve_polygon(nm) for nm in {op["polygon"] for s in L["scenarios"].values() for op in s["ops"] if "polygon" in op}}
    for nm, pg in polys.items():
        if pg.intersection(wbox).area / pg.area < 0.99:
            raise ValueError(f"polygon {nm} lies outside the study window")

    all_rows, plans_out, maps = [], {}, {}
    for sname, sc in L["scenarios"].items():
        sw = sc["sweep"]
        for val in sw["values"]:
            ops = copy.deepcopy(sc["ops"])
            ops[0][sw["param"]] = val
            for op in ops:
                if "alignment" in op:
                    op["line_xy"] = lines[op.pop("alignment")]
                if "polygon" in op:
                    op["polygon_xy"] = polys[op.pop("polygon")]
            res, info, mp = study.evaluate(ops, reps_per_set=reps, seed=L["model"]["seed"])
            if ops[0]["type"] == "plant":
                ha_edited = res.ha_tree_gain
            else:
                ha_edited = pd.Series(next(p["area_ha"] for p in info["plans"] if p["kind"] in ("road", "reclass")), index=res.index)
            res["person_degc_per_ha_edited"] = res.person_degc / ha_edited.where(ha_edited > 0)
            res.insert(0, "variant", val); res.insert(0, "scenario", sname)
            all_rows.append(res)
            plans_out[f"{sname}__{lr.vkey(val)}"] = info["plans"]
            for k, v in mp.items():
                maps[f"{sname}__{lr.vkey(val)}__{k}"] = v
            s_ = res.zone_mean_dlst_c
            log(f"{sname:22s} {sw['param']}={str(val):<6} zone mean {s_.mean():+.4f} C [{s_.quantile(.05):+.4f}, {s_.quantile(.95):+.4f}]  "
                f"pop-weighted {res.popw_dlst_c.mean():+.4f} C  plan {info['plans'][0]}")
    runs = pd.concat(all_rows, ignore_index=True)
    summ = lr.summarise(runs, ["scenario", "variant"], lr.METRICS + ["person_degc_per_ha_edited"])
    summ.insert(2, "label", summ.scenario.map({k: v.get("label", k) for k, v in L["scenarios"].items()}))
    summ.insert(3, "param", summ.scenario.map({k: v["sweep"]["param"] for k, v in L["scenarios"].items()}))
    summ.to_csv(out / "lulc_scenario_summary.csv", index=False)
    runs.to_csv(out / "lulc_scenario_runs.csv", index=False)
    study.coef.to_csv(out / "lulc_model_coefficients.csv", index=False)
    study.cv.to_csv(out / "lulc_model_cv.csv", index=False)
    json.dump(plans_out, open(out / "lulc_scenario_plans.json", "w"), indent=1)
    np.savez_compressed(out / "lulc_maps.npz", **maps)
    geom = {"t30": [study.t30.a, study.t30.c, study.t30.e, study.t30.f], "shape30": list(study.lst.shape), "centre": list(centre),
            "zone_radius_m": L["study"]["zone_radius_m"], "siting_radius_m": L["study"]["siting_radius_m"],
            "streets": [[list(map(list, part.coords)) for part in (s.geoms if hasattr(s, "geoms") else [s])] for s in study.streets.geometry],
            "alignments": {k: list(map(list, v.coords)) for k, v in lines.items()}, "crs": crs.to_string()}
    json.dump(geom, open(out / "lulc_geometry.json", "w"))
    pickle.dump(study, open(out / "lulc_study.pkl", "wb"))
    json.dump({"methodology": "observational land-cover to surface-temperature model; before/after from edited 10 m land-cover shares",
               "settings": L, "reps_per_set": reps, "hot_threshold_c": study.hot_thr,
               "n_replicates_per_scenario_size": int(summ.n_replicates.iloc[0]), "model_problems": study.problems, "model_warnings": study.warnings,
               "population_weighted": study.has_pop,
               "not_modelled": ["air temperature and thermal comfort", "indoor heat and cooling demand", "traffic and anthropogenic heat",
                                "tree growth time, species, irrigation, soil", "construction phase", "pedestrian shade (see walking tab)",
                                "cost", "population relocation (displaced people are counted, not moved)"]},
              open(out / "lulc_assumptions.json", "w"), indent=2, default=str)
    return runs, summ
