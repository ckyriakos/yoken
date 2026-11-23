"""Plain-language explanations of every layer, for non-technical users (info box, legends, API).

Each entry: what = what you see; proxy = what it stands for in the real world; read = how to read high/low values;
when = the exact period the data describes; unit = unit of sampled values; source.
Satellite periods are read from the configs so they always match the data actually used."""
from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

import yaml

from . import layers as LY


def _fmt(days):
    days = sorted(days)
    if not days:
        return ""
    a, b = days[0], days[-1]
    return f"{a:%-d %b}–{b:%-d %b %Y}" if a.year == b.year else f"{a:%-d %b %Y}–{b:%-d %b %Y}"


@lru_cache(maxsize=1)
def periods() -> dict:
    """Exact image dates behind 'summer' and 'wet season'."""
    out = {}
    try:
        cfg = yaml.safe_load(open(LY.ROOT / "config.yaml"))
        roles = set(cfg.get("composite", {}).get("roles_used", []))
        ls = [datetime.strptime(s["landsat"].split("_")[3], "%Y%m%d") for s in cfg["scenes"] if s.get("role") in roles]
        s2 = [datetime.strptime(s["s2"].split("_")[2][:8], "%Y%m%d") for s in cfg["scenes"] if s.get("role") in roles]
        out["summer_lst"] = f"Summer 2026: median of {len(ls)} cloud-free Landsat 8/9 images, {_fmt(ls)}, taken around midday local time"
        out["summer_ndvi"] = f"Summer 2026: median of {len(s2)} Sentinel-2 images, {_fmt(s2)}"
    except Exception:
        out["summer_lst"] = out["summer_ndvi"] = "Summer 2026 (June–September)"
    try:
        sp = json.loads((LY.OUT / "cases" / "flood" / "spectral.json").read_text())
        d = [datetime.fromisoformat(s["datetime"][:10]) for s in sp["scenes"]]
        out["wet"] = (f"Wet season 2026: median of {len(d)} cloud-free Sentinel-2 images, {_fmt(d)} "
                      "(winter–spring, when soils are wettest and puddles/streams most visible)")
    except Exception:
        out["wet"] = "Wet season 2026 (January–April)"
    try:
        lc = json.loads((LY.OUT / "cases" / "kallimarmaro" / "landcover" / "landcover.json").read_text())
        for k, v in lc["periods"].items():
            d = [datetime.fromisoformat(s["datetime"][:10]) for s in v["scenes"]]
            out[f"olgas_{k}"] = f"Summer {v['year']}: median of {len(d)} Sentinel-2 images, {_fmt(d)}"
    except Exception:
        pass
    return out


def _e(what, proxy="", read="", when="", unit="", source=""):
    return {"what": what, "proxy": proxy, "read": read, "when": when, "unit": unit, "source": source}


def about(layer_id: str) -> dict:
    if "__" in layer_id:                                  # user-chosen Sentinel-2 period: same meaning, its own dates
        base, key = layer_id.split("__", 1)
        e = dict(about(base))
        try:
            from .services import flood
            per = next((p for p in flood.periods() if p["key"] == key), None)
            if per:
                d = sorted(x["datetime"][:10] for x in per["scenes"])
                e["when"] = (f"Median of {len(d)} cloud-free Sentinel-2 images between {per['start']} and {per['end']} "
                             f"(image dates {d[0]} … {d[-1]}). Compare with the wet-season layers: dry-season values are lower.")
        except Exception:
            pass
        return e
    P = periods()
    ndvi_read = "Above 0.5: dense trees or lawns. 0.2–0.5: patchy or dry vegetation. Below 0.15: roofs, roads, bare ground; below 0: water."
    E = {
        "districts": _e("The 7 municipal districts (Δημοτικές Κοινότητες) of Athens.", source="City of Athens GIS"),
        "neighbourhoods": _e("The 144 official neighbourhoods; most statistics in this platform are summarised per neighbourhood.",
                             source="City of Athens GIS"),
        "roads": _e("Street centrelines of the municipal road network.", source="City of Athens GIS"),
        "landuse": _e("The statutory land-use (zoning) plan: what each block is allowed to be used for.",
                      proxy="Intended use, not necessarily current use.", source="City of Athens GIS (EPSG:2100)"),
        "parks": _e("Parks mapped in OpenStreetMap.", source="OpenStreetMap via Athens GIS repository"),
        "gardens": _e("Gardens mapped in OpenStreetMap (incl. private and institutional gardens).", source="OpenStreetMap"),
        "pocket_parks": _e("The City's small 'pocket parks' on former vacant plots.", source="City of Athens"),
        "trees": _e("Density of individually mapped trees (inventory points), shown as a heatmap.",
                    proxy="Where the tree inventory is dense; gaps can mean 'not surveyed' as well as 'no trees'.", source="Attica tree inventory"),
        "street_trees": _e("Streets lined with trees, from satellite-based mapping.", source="Copernicus Urban Atlas Street Tree Layer 2021"),
        "tree_density": _e("Share of each 10 m square covered by tree crowns.", proxy="Shade and cooling potential from trees.",
                           read="0% = no canopy, 100% = fully covered.", when="2024", unit="%", source="Copernicus HRL Tree Cover Density 2024"),
        "leaf_type": _e("Whether the dominant trees are broadleaved (e.g. plane, bitter orange) or coniferous (e.g. pine).",
                        proxy="Broadleaved trees give denser summer shade.", when="2024", source="Copernicus HRL Dominant Leaf Type 2024"),
        "bus_stops": _e("OASA bus and trolley stops.", source="OASA"),
        "traffic_lights": _e("Signalised junctions and crossings.", source="City of Athens"),
        "ai_cameras": _e("AI traffic cameras (the source marks this list as outdated).", source="City of Athens"),
        "traffic_detectors": _e("Region of Attica loop detectors buried in the road: they count passing vehicles.",
                                proxy="Traffic volume on the main roads.", read="Vehicles per hour in one direction; speed in km/h.",
                                when="Latest reading (updated every 90 s); history is collected by this platform's logger.",
                                unit="veh/h", source="data.gov.gr, Region of Attica (CC BY 4.0)"),
        "accidents": _e("Police-recorded traffic accidents, shown as a heatmap.", when="2023–2025", source="Hellenic Police"),
        "flood_50": _e("Area expected to flood in a storm that happens on average once every 50 years (2% chance each year).",
                       read="Darkest blue = most frequent flooding.", source="Flood Risk Management Plan (EU Floods Directive)"),
        "flood_100": _e("Area expected to flood in a once-in-100-years storm (1% chance each year).",
                        source="Flood Risk Management Plan (EU Floods Directive)"),
        "flood_1000": _e("Area expected to flood in an extreme, once-in-1000-years storm (0.1% chance each year).",
                         source="Flood Risk Management Plan (EU Floods Directive)"),
        "rivers": _e("Rivers and streams, including those now running in underground culverts (e.g. Ilisos, Kifisos tributaries).",
                     proxy="Where water naturally collects during heavy rain.", source="Athens GIS repository"),
        "soil_groups": _e("Soil types (World Reference Base groups).", proxy="How easily rain soaks in: clay soils (Vertisols) shed water fast.",
                          source="European Soil Database"),
        "seismic": _e("Official seismic hazard zones for building design.", source="EAK (Greek Seismic Code)"),
        "earthquakes": _e("Recorded earthquakes in Attica.", when="2006–2026", source="NOA / EMSC"),
        "building_height": _e("Average building height in each 10 m square.", proxy="Density and shading: tall blocks shade streets but trap heat.",
                              unit="m", when="2021", source="Copernicus Building Height 2021"),
        "population_density": _e("Residents per km² on a 1 km grid.", unit="people/km²", when="2021 census", source="Eurostat / ELSTAT"),
        "buildings": _e("Building footprints with rooftop solar potential.", source="Athens GIS repository"),
        "lst": _e("Land Surface Temperature: how hot the ground and roofs are, measured by satellite from their heat radiation.",
                  proxy="Urban heat island: where surfaces bake on hot days. Surface temperature is higher than the air temperature people feel.",
                  read="Yellow/white = hottest (asphalt, flat roofs), dark purple = coolest (parks, shaded streets).",
                  when=P.get("summer_lst", ""), unit="°C", source="USGS Landsat Collection 2, our median composite"),
        "hotspots": _e("Statistically significant clusters of hot surfaces (hot area surrounded by hot area).",
                       proxy="Heat island cores, not isolated hot roofs.", when=P.get("summer_lst", ""), source="Getis-Ord Gi* on the LST map"),
        "ndvi": _e("NDVI (vegetation index): how green and leafy the surface is, from how plants reflect red vs near-infrared light.",
                   proxy="Amount of healthy vegetation, i.e. natural cooling and shade.", read=ndvi_read, when=P.get("summer_ndvi", ""),
                   source="Copernicus Sentinel-2"),
        "priority": _e("Heat priority per neighbourhood: hot surfaces on the hottest days combined with little greenery.",
                       read="Darker red = higher priority for cooling measures.", when=P.get("summer_lst", ""), source="This platform"),
        "s2_ndwi": _e("NDWI (water index): compares green and near-infrared light; water and very wet surfaces stand out.",
                      proxy="Open water, waterlogged ground and very moist vegetation.",
                      read="Above 0 (blue): water or very wet surface. Below −0.3 (red): dry vegetation or dry ground.",
                      when=P.get("wet", ""), source="Copernicus Sentinel-2"),
        "s2_mndwi": _e("MNDWI (modified water index): compares green and short-wave infrared light; better than NDWI at separating "
                       "water from roofs and roads.", proxy="Surface water and ponding; useful to spot where water stands after rain.",
                       read="Above 0 (blue): water. Around −0.2 to 0: wet soil or shadows. Below −0.4 (red): dry, built or bare.",
                       when=P.get("wet", ""), source="Copernicus Sentinel-2"),
        "s2_ndbi": _e("NDBI (built-up index): compares short-wave infrared and near-infrared light; roofs, asphalt and concrete stand out.",
                      proxy="Sealed (impervious) surfaces, where rain cannot soak in and runs off.",
                      read="Above 0 (dark orange): built and sealed. Below 0: vegetation or water.", when=P.get("wet", ""),
                      source="Copernicus Sentinel-2"),
        "s2_ndvi": _e("NDVI in the wet season: greenness when grass and shrubs are at their peak.",
                      proxy="Vegetation that slows and absorbs rainwater.", read=ndvi_read, when=P.get("wet", ""), source="Copernicus Sentinel-2"),
        "runoff": _e("Runoff susceptibility: a screening score of where heavy rain is likely to run off rather than soak in.",
                     proxy="0.4 × sealed surface (NDBI) + 0.2 × lack of vegetation + 0.2 × clay-rich soil + 0.2 × closeness to a stream.",
                     read="0 = water soaks in, 1 = water runs off fast. A screening map, not a flood model.", when=P.get("wet", ""),
                     source="This platform"),
        "olgas_ndvi_pre_works": _e("Greenness (NDVI) around Vas. Olgas before the 2020 closure.", read=ndvi_read,
                                   when=P.get("olgas_pre_works", "Summer 2019"), source="Copernicus Sentinel-2"),
        "olgas_ndvi_works": _e("Greenness (NDVI) around Vas. Olgas during the works.", read=ndvi_read,
                               when=P.get("olgas_works", "Summer 2025"), source="Copernicus Sentinel-2"),
        "olgas_ndvi_after": _e("Greenness (NDVI) around Vas. Olgas in the first summer after the 2026 planting.", read=ndvi_read,
                               when=P.get("olgas_after", "Summer 2026"), source="Copernicus Sentinel-2"),
        "olgas_ndvi_change": _e("Change in greenness (NDVI) from before the works (2019) to after (2026).",
                                read="Green = greener now, brown = less green now.", source="Copernicus Sentinel-2"),
    }
    E["env_sensors"] = _e("City of Athens street-level stations measuring air temperature, humidity and air quality every hour.",
                          proxy="The heat and air people actually breathe, unlike satellite surface temperature which measures hot roofs and roads.",
                          read="Colour = average air temperature on summer afternoons (Jul–Aug, 12:00–17:00). Click a station for its latest "
                               "reading, a 7-day temperature line and its neighbourhood's satellite surface temperature for comparison.",
                          when="Hourly, 1 Jan – 20 Nov 2026", unit="°C", source="opendata.cityofathens.gr (Smart Neighbourhood, env-all-stations)")
    E["area_types"] = _e("What kind of place each neighbourhood is (dense residential, mixed, civic, industrial…), from zoning and "
                         "building form, and which cooling measures fit it.",
                         read="Click a neighbourhood: its type, how hot and green it is compared with the rest of Athens, and the measures "
                              "that fit. ● = simulated on the Heat page, ◐ = approximated, ○ = needs data first.",
                         source="This platform: zoning plan + building footprints + summer satellite tiers")
    e = dict(E.get(layer_id, _e(LY.BY_ID[layer_id].description if layer_id in LY.BY_ID else "")))
    base = layer_id.split("__")[0]
    if base in SCALE:
        e["scale"] = SCALE[base]
    if base in METHOD:
        e["method"] = METHOD[base]
    return e


# plain-language ends of each colour bar (low value, high value)
SCALE = {
    "lst": ("cooler surfaces (parks, shade)", "hottest surfaces (asphalt, flat roofs)"),
    "ndvi": ("roofs, roads, bare ground", "dense trees and lawns"), "s2_ndvi": ("roofs, roads, bare ground", "dense trees and lawns"),
    "olgas_ndvi_pre_works": ("bare / built", "dense vegetation"), "olgas_ndvi_works": ("bare / built", "dense vegetation"),
    "olgas_ndvi_after": ("bare / built", "dense vegetation"), "olgas_ndvi_change": ("less green than 2019", "greener than 2019"),
    "s2_ndwi": ("dry ground, dry plants", "water, very wet surfaces"), "s2_mndwi": ("dry, built or bare", "standing water"),
    "s2_ndbi": ("vegetation, water", "roofs, asphalt, concrete"), "runoff": ("rain soaks in", "rain runs off fast"),
    "tree_density": ("no tree canopy", "fully covered by canopy"), "building_height": ("low-rise", "tall blocks"),
    "population_density": ("few residents", "many residents"), "priority": ("lower priority", "higher heat priority"),
    "env_sensors": ("cooler afternoons", "hotter afternoons"),
}
METHOD = {
    "ndvi": "(near-infrared − red) / (near-infrared + red). Plants reflect near-infrared strongly and absorb red.",
    "s2_ndvi": "(near-infrared − red) / (near-infrared + red). Plants reflect near-infrared strongly and absorb red.",
    "s2_ndwi": "(green − near-infrared) / (green + near-infrared), McFeeters 1996. Water absorbs near-infrared.",
    "s2_mndwi": "(green − short-wave infrared) / (green + short-wave infrared), Xu 2006. Separates water from built surfaces better than NDWI.",
    "s2_ndbi": "(short-wave infrared − near-infrared) / (short-wave infrared + near-infrared). Built materials reflect more short-wave infrared.",
    "lst": "Landsat thermal band (10.9 µm) converted to surface temperature by USGS (emissivity-corrected); median of cloud-free pixels.",
    "runoff": "0.4 × built-up (NDBI) + 0.2 × (1 − vegetation) + 0.2 × soil runoff class + 0.2 × closeness to a stream (300 m decay).",
    "hotspots": "Getis-Ord Gi* on the surface temperature map with false-discovery-rate correction.",
    "priority": "Hot-day surface temperature (P90) and low greenness, ranked across eligible neighbourhoods with bootstrap uncertainty.",
}


# readable attribute names for the info box
FIELD_LABELS = {
    "name": "Name", "name_en": "Name", "synoikia": "Neighbourhood", "geitonia": "Area", "dk": "District", "street_gr": "Street",
    "street_en": "Street", "ypomnima": "Zoning", "xriseis": "Permitted uses", "shapearea_m": "Area (m²)", "descr": "Description",
    "streetname": "Street", "species": "Species", "genus": "Genus", "leaf_type": "Leaf type", "stop_id": "Stop id", "route_ref": "Routes",
    "Type": "Type", "Cause": "Cause", "Date": "Date", "period": "Return period (years)", "name_en_": "Name", "SoilGroup": "Soil group",
    "Zone": "Seismic zone", "mag": "Magnitude", "place": "Place", "Height": "Height (m)", "IRR_GLOBAL": "Solar irradiation",
    "N_PANELS": "Possible solar panels", "neighbourhood_name": "Neighbourhood", "neighbourhood_name_en": "Neighbourhood",
    "lst_p90_c": "Hot-day surface temp. (P90, °C)", "ndvi_mean": "Greenness (mean NDVI)", "priority_score": "Heat priority score",
    "priority_rank": "Priority rank (1 = highest)", "hotspot_share_pct": "Area in heat hotspot (%)", "id": "Id", "odos": "Street",
    "description": "Position", "last_vph": "Last count (veh/h)", "last_speed": "Last speed (km/h)", "last_time": "Reading time (UTC)",
    "observed_vph": "Detector count (veh/h)",
    "area_type_label": "Area type", "area_type_why": "Why this type", "heat": "Heat vs rest of Athens", "greenery": "Greenery vs rest of Athens",
    "address": "Address", "neighbourhood": "Neighbourhood", "district": "District", "location": "Position",
    "last_time_local": "Latest reading", "temp_c": "Air temperature (°C)", "humidity_pct": "Humidity (%)", "aqi": "Air quality index",
    "pm2_5": "PM2.5 (µg/m³)", "no2": "NO₂ (µg/m³)", "o3": "O₃ (µg/m³)", "summer_pm_temp_c": "Summer afternoons, mean air temp. (°C)",
    "summer_pm_temp_max_c": "Summer afternoons, max air temp. (°C)", "nb_lst_p90_c": "Neighbourhood surface temp., hot days (°C)", "model_vph": "Model (veh/h)", "ratio": "Detector ÷ model",
}
SKIP_FIELDS = {"station_id", "eligible", "n_measures", "neighbourhood_id", "OBJECTID", "FID", "rwb_id", "basinid_fd", "geit_thesm", "shape_area", "layer", "zone"}
