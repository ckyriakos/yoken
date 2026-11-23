"""Catalogue of every map layer the demo and the API expose.

Each layer has a stable id (used in API URLs and by the future chat agent), a group for the UI, a source path, a kind and a style.
Kinds:
  polygon / line / point  -> vector, served as GeoJSON (EPSG:4326, clipped to the Athens municipality, simplified)
  heat                    -> point vector drawn as a heatmap (too many points for markers)
  raster                  -> single-band raster, pre-rendered to a coloured PNG overlay (EPSG:3857) and served as XYZ tiles
Paths are relative to the project root (ROOT)."""
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIS = "gis_Data"
REPO = f"{GIS}/AthensGISRepository_"
OUT = ROOT / "data" / "outputs"
DEMO = OUT / "demo"
LAYER_DIR = DEMO / "layers"

# Athens municipality (with margin) and the Kallimarmaro case-study centre, WGS84
ATHENS_CENTER = (37.9838, 23.7275)
KALLIMARMARO = (37.9685, 23.7411)


@dataclass
class Layer:
    id: str
    title: str
    group: str
    path: str
    kind: str
    description: str = ""
    style: dict = field(default_factory=dict)
    # raster rendering: colormap name, value range (None = 2-98 percentile), categorical labels
    cmap: str = "viridis"
    vrange: tuple | None = None
    categories: dict | None = None
    # vector preparation
    simplify_m: float = 0.0
    keep: tuple = ()             # attribute columns to keep (empty = all)
    weight: str | None = None    # heatmap weight column
    default_on: bool = False
    clip: bool = True            # clip to the Athens municipality

    def to_dict(self):
        d = asdict(self)
        d["prepared"] = prepared_path(self).exists()
        return d


L = Layer
LAYERS: list[Layer] = [
    # --- Base
    L("districts", "Municipal districts", "Base", f"{GIS}/athens_districts.geojson", "polygon", "7 municipal districts (53 sub-units)",
      {"color": "#444", "weight": 1.2, "fillOpacity": 0.0}, keep=("name", "name_en", "dk"), default_on=True, clip=False),
    L("neighbourhoods", "Neighbourhoods", "Base", f"{GIS}/athens_neighbourhoods.geojson", "polygon", "144 neighbourhoods",
      {"color": "#666", "weight": 0.6, "fillOpacity": 0.0, "dashArray": "3"}, keep=("id", "synoikia", "name_en", "dk"), clip=False),
    L("roads", "Road network", "Base", f"{GIS}/athens_roads.geojson", "line", "Municipal road centrelines",
      {"color": "#8a8a8a", "weight": 0.6, "opacity": 0.7}, simplify_m=1.5, keep=("street_gr", "street_en")),
    L("landuse", "Zoning / land use", "Base", f"{GIS}/landuse.geojson", "polygon", "Statutory land-use plan (EPSG:2100 source)",
      {"color": "#b07aa1", "weight": 0.3, "fillOpacity": 0.25}, simplify_m=2, keep=("ypomnima", "xriseis")),
    # --- Green
    L("parks", "Parks", "Green", f"{REPO}Parks/Parks.geojson", "polygon", "OSM leisure=park",
      {"color": "#2e7d32", "weight": 0.8, "fillColor": "#66bb6a", "fillOpacity": 0.5}, simplify_m=1, keep=("name", "shapearea_m"), default_on=True),
    L("gardens", "Gardens", "Green", f"{REPO}Gardens/Gardens.geojson", "polygon", "OSM leisure=garden",
      {"color": "#558b2f", "weight": 0.6, "fillColor": "#9ccc65", "fillOpacity": 0.5}, simplify_m=1, keep=("name",)),
    L("pocket_parks", "Pocket parks", "Green", f"{GIS}/athens_pocket_parks.geojson", "point", "City of Athens pocket parks",
      {"color": "#1b5e20", "radius": 6, "fillColor": "#43a047", "fillOpacity": 0.9}),
    L("trees", "Trees (points)", "Green", f"{REPO}Trees (Points)/attica_trees.geojson", "heat", "Tree inventory, 30k points",
      keep=("species", "genus", "leaf_type")),
    L("street_trees", "Street trees (2021)", "Green", f"{REPO}Street Trees (2021)/street_trees.tif", "raster",
      "Urban Atlas street-tree mask, 10 m", cmap="Greens", vrange=(0, 1), categories={1: "Street trees"}),
    L("tree_density", "Tree cover density (2024)", "Green", f"{REPO}Tree Cover Density (2024)/tree_density.tif", "raster",
      "Copernicus tree cover 0-100 %, 10 m", cmap="YlGn", vrange=(0, 100)),
    L("leaf_type", "Dominant leaf type (2024)", "Green", f"{REPO}Dominant Leaf Type (2024)/leaf_type.tif", "raster",
      "1 = broadleaved, 2 = coniferous", categories={1: "Broadleaved", 2: "Coniferous"}, vrange=(1, 2), cmap="Set2"),
    # --- Mobility
    L("bus_stops", "OASA bus stops", "Mobility", f"{REPO}OASA Bus Stops/AthensBusStops.geojson", "point", "Bus/trolley stops",
      {"color": "#1565c0", "radius": 2.5, "fillColor": "#42a5f5", "fillOpacity": 0.9, "weight": 0.5}, keep=("stop_id", "name", "route_ref")),
    L("traffic_lights", "Traffic lights", "Mobility", f"{REPO}Traffic Lights/TrLights.geojson", "point", "Signalised junctions",
      {"color": "#e65100", "radius": 2, "fillColor": "#ff9800", "fillOpacity": 0.9, "weight": 0.5}),
    L("ai_cameras", "AI traffic cameras", "Mobility", f"{REPO}AI Traffic Cameras/AICameras.geojson", "point", "Source flags as outdated",
      {"color": "#6a1b9a", "radius": 6, "fillColor": "#ab47bc", "fillOpacity": 0.9}),
    L("env_sensors", "Air temperature & air quality stations", "Sensors", "data/raw/environment/stations.geojson", "point",
      "City of Athens Smart-Neighbourhood stations: hourly temperature, humidity, AQI, PM, NO2, O3",
      {"color": "#1d2939", "radius": 7, "weight": 1.2, "fillOpacity": 0.95, "choropleth": "summer_pm_temp_c", "cmap": "YlOrRd",
       "vrange": (30, 38)}, keep=(), clip=False, default_on=False),
    L("traffic_detectors", "Traffic detectors (Attica, live)", "Sensors", "data/raw/traffic/stations.geojson", "point",
      "Region of Attica loop detectors: last veh/h and speed (data.gov.gr, CC BY 4.0)",
      {"color": "#1d2939", "radius": 5, "fillColor": "#fdb022", "fillOpacity": 0.95, "weight": 1},
      keep=("id", "odos", "description", "last_vph", "last_speed", "last_time")),
    L("accidents", "Traffic accidents 2023-25", "Mobility", f"{REPO}Traffic Accidents 2023-2025/TrAccidents.geojson", "heat",
      "14k police-recorded accidents", keep=("Type", "Cause", "Date")),
    # --- Hazards
    L("flood_50", "Flood zone, 50-yr", "Hazards", f"{REPO}Flood Risk Zones 50T/Floods50T.geojson", "polygon", "Flood Directive hazard zone T=50",
      {"color": "#0d47a1", "weight": 0.4, "fillColor": "#1e88e5", "fillOpacity": 0.55}, simplify_m=3, keep=("period",)),
    L("flood_100", "Flood zone, 100-yr", "Hazards", f"{REPO}Flood Risk Zones 100T/Floods100T.geojson", "polygon", "Flood Directive hazard zone T=100",
      {"color": "#1565c0", "weight": 0.4, "fillColor": "#42a5f5", "fillOpacity": 0.45}, simplify_m=3, keep=("period",)),
    L("flood_1000", "Flood zone, 1000-yr", "Hazards", f"{REPO}Flood Risk Zones 1000T/Floods1000T.geojson", "polygon", "Flood Directive hazard zone T=1000",
      {"color": "#1976d2", "weight": 0.4, "fillColor": "#90caf9", "fillOpacity": 0.35}, simplify_m=3, keep=("period",)),
    L("rivers", "Rivers and streams", "Hazards", f"{REPO}Rivers and Streams/Rivers.geojson", "line", "Incl. culverted streams",
      {"color": "#0277bd", "weight": 1.6, "opacity": 0.9}, simplify_m=3, keep=("name_en",)),
    L("soil_groups", "Soil groups", "Hazards", f"{REPO}Soil Groups/SoilGroups.geojson", "polygon", "WRB soil groups",
      {"color": "#795548", "weight": 0.3, "fillColor": "#a1887f", "fillOpacity": 0.3}, simplify_m=5, keep=("SoilGroup",)),
    L("seismic", "Seismic hazard zones", "Hazards", f"{REPO}Seismic Hazard Zones/Seismiczones.geojson", "polygon", "EAK zones",
      {"color": "#c62828", "weight": 0.8, "fillColor": "#ef5350", "fillOpacity": 0.15}, simplify_m=10, keep=("Zone",), clip=False),
    L("earthquakes", "Earthquakes 2006-2026", "Hazards", f"{REPO}Earthquakes Attica 2006-2026/EarthquakesAttica2006-2026.geojson", "point",
      "Attica, all magnitudes", {"color": "#b71c1c", "radius": 4, "fillColor": "#e53935", "fillOpacity": 0.7}, keep=("mag", "place", "Date"), clip=False),
    # --- Buildings and people
    L("building_height", "Building height (2021)", "Buildings & people", f"{REPO}Buildings height (2021)/Building_height.tif", "raster",
      "Copernicus building height, 10 m", cmap="magma", vrange=(0, 30)),
    L("population_density", "Population density (2021)", "Buildings & people", "data/raw/gis/population_density_2021.tif", "raster",
      "Eurostat 1 km grid, people/km²", cmap="OrRd", vrange=(0, 30000), clip=False),
    L("buildings", "Buildings (solar potential)", "Buildings & people", f"{GIS}/solar_efficiency.geojson", "polygon",
      "66k footprints; served by bbox only", {"color": "#5d4037", "weight": 0.3, "fillColor": "#ffb74d", "fillOpacity": 0.6},
      simplify_m=0.5, keep=("Height", "Type", "IRR_GLOBAL", "N_PANELS")),
    # --- Heat (pipeline outputs)
    L("lst", "Surface temperature (summer 2026)", "Heat", "data/outputs/lst_composite.tif", "raster",
      "Landsat 8/9 median of 7 scenes, 29 Jun – 1 Sep 2026, °C", cmap="inferno", vrange=(35, 55), default_on=True, clip=False),
    L("hotspots", "Heat hotspots (summer 2026)", "Heat", "data/outputs/hotspots.tif", "raster", "Getis-Ord Gi* significant hot cells",
      cmap="Reds", vrange=(0, 1), categories={1: "Hotspot"}, clip=False),
    L("ndvi", "Greenness, NDVI (summer 2026)", "Heat", "data/outputs/ndvi_composite.tif", "raster", "Sentinel-2 median NDVI",
      cmap="RdYlGn", vrange=(-0.1, 0.7), clip=False),
    L("area_types", "Area type & fitting measures", "Heat", "data/outputs/demo/area_types.geojson", "polygon",
      "What kind of place each neighbourhood is, and which cooling measures fit it",
      {"color": "#333", "weight": 0.5, "fillOpacity": 0.55, "categorical": "area_type_label"}, keep=(), clip=False),
    L("priority", "Heat priority score", "Heat", "data/outputs/neighbourhood_indicators.geojson", "polygon",
      "Neighbourhood heat priority (LST P90 + low NDVI)",
      {"color": "#333", "weight": 0.5, "fillOpacity": 0.6, "choropleth": "priority_score", "cmap": "YlOrRd"},
      keep=("neighbourhood_id", "neighbourhood_name", "neighbourhood_name_en", "lst_p90_c", "ndvi_mean", "priority_score", "priority_rank",
            "hotspot_share_pct"), clip=False),
]
BY_ID = {l.id: l for l in LAYERS}
GROUPS = list(dict.fromkeys(l.group for l in LAYERS))
# layers too heavy to send whole: only served for a bbox
BBOX_ONLY = {"buildings"}


def prepared_path(layer: Layer) -> Path:
    if layer.kind == "raster":
        return LAYER_DIR / f"{layer.id}.png"
    return LAYER_DIR / f"{layer.id}.geojson"


def get(layer_id: str) -> Layer:
    if layer_id not in BY_ID:
        raise KeyError(f"unknown layer '{layer_id}'. Known: {', '.join(BY_ID)}")
    return BY_ID[layer_id]
