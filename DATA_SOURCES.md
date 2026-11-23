# Data sources

Links marked *(canonical page)* point to the public page of a source the project names without a URL; they have not been opened to verify. "Not stated" = no licence given in the project or the layer's source note.

## City of Athens open data & GIS

| Source | Used for | Link | Licence |
|---|---|---|---|
| City of Athens GIS repository (`gis_Data/`, `data/raw/gis/`) | Districts, neighbourhoods, roads, zoning/land use, pocket parks, building footprints with solar potential | no URL in the project | Not stated |
| City of Athens open data: environmental stations, hourly (`env-all-stations`) | Air temperature and air quality readings; sensor validation; `env_sensors` layer | https://opendata.cityofathens.gr/dataset/env-all-stations (dump: https://opendata.cityofathens.gr/datastore/dump/a84cabca-bd0c-434d-8bee-21c082183442?format=csv) | Not stated |
| City of Athens open data: station registry (`env-stations-registry`) | Station addresses (geocoded once) | https://opendata.cityofathens.gr/api/3/action/datastore_search?resource_id=47ba486c-4ecf-497e-892b-4e47c53f422a&limit=500 | Not stated |
| Copernicus Urban Atlas Street Tree Layer 2021 | `street_trees` | https://land.copernicus.eu/en/products/urban-atlas *(canonical page)* | Copernicus free and open policy |
| Copernicus HRL Tree Cover Density 2024 | `tree_density` | https://land.copernicus.eu/en/products/high-resolution-layer-tree-cover-density *(canonical page)* | Copernicus free and open policy |
| Copernicus HRL Dominant Leaf Type 2024 | `leaf_type` | https://land.copernicus.eu/en/products/high-resolution-layer-dominant-leaf-type *(canonical page)* | Copernicus free and open policy |
| Copernicus building height 2021 (10 m) | `building_height` | https://land.copernicus.eu *(canonical page)* | Copernicus free and open policy |
| OpenStreetMap (via the GIS repository) | Parks, gardens, traffic lights, street centrelines (`Streets_Center`) | https://www.openstreetmap.org *(canonical page)* | ODbL |
| Tree inventory (data.gov.gr "Dendra Kipou") + OSM trees | `trees` points | https://data.gov.gr *(canonical page)* | ODbL 1.0 (OSM part) |
| OASA | Bus stops; bus-stop access | https://www.oasa.gr *(canonical page)* | Not stated |
| Ministry of Digital Governance | AI traffic cameras (marked outdated by the source) | https://mindigital.gr *(canonical page)* | Not stated |
| Road Traffic Collision Map (gov.gr) | Traffic accidents 2023–2025 | https://www.gov.gr *(canonical page)* | Not stated |
| Ministry of Environment and Energy, flood hazard maps | Flood zones 50/100/1000-yr; flood network impact | https://floods.ypeka.gr *(canonical page)* | Not stated |
| Ministry of Environment and Energy | Rivers and streams | https://ypen.gov.gr *(canonical page)* | Not stated |
| SoilGrids | Soil groups (WRB) | https://soilgrids.org *(canonical page)* | CC BY 4.0 (SoilGrids) |
| EAK (Greek seismic code) | Seismic hazard zones | https://www.oasp.gr *(canonical page)* | Not stated |
| USGS | Earthquakes M2.5+, 2006–2026 | https://earthquake.usgs.gov/earthquakes/search/ *(canonical page)* | Not stated |
| ELSTAT 2021 census | Residents per district (population model) | https://www.statistics.gr *(canonical page)* | Not stated |
| Eurostat 2021 population grid | `population_density` (1 km) | https://ec.europa.eu/eurostat/web/gisco/geodata/population-distribution/population-grids *(canonical page)* | Not stated |
| Hellenic Cadastre (KAEK) | Present in `data/raw/gis`, not used by the code | https://www.ktimatologio.gr *(canonical page)* | Not stated |

## Regional / national open data

| Source | Used for | Link | Licence |
|---|---|---|---|
| Region of Attica traffic detectors, live | `traffic_detectors` layer; logger; Vas. Olgas calibration | https://data.gov.gr/dataset/attica-region-traffic-data (feed: https://loader.data.gov.gr/data/attica-region-traffic-data) | CC BY 4.0 |
| Region of Attica traffic history (`road_traffic_attica`) | Hourly counts 22–23 Nov 2025 ("before" calibration) | https://data.gov.gr/dataset/road_traffic_attica | CC BY 4.0 |
| Apps4Athens data-logger (GitHub Actions artifacts) | Weekday-peak detector readings, 26 Oct–20 Nov 2026 ("after" calibration) | https://github.com/nickoulos/apps4athens-data-logger | MIT (code), CC BY 4.0 (data) |

## Satellite imagery & Earth observation

| Source | Used for | Link | Licence / credit |
|---|---|---|---|
| Planetary Computer STAC: Landsat Collection 2 L2 | Land surface temperature; Landsat timelapse | https://planetarycomputer.microsoft.com/api/stac/v1 | "Landsat imagery courtesy of NASA/USGS" |
| Planetary Computer: Sentinel-2 L2A | NDVI; flood and land-cover indices; custom periods; timelapse | same | "Contains modified Copernicus Sentinel data" |
| Planetary Computer: ESA WorldCover 2020/2021 | Reference labels for the land-cover classifier | same | Not stated |
| Planetary Computer: Impact Observatory LULC | Optional reference land cover (`lulc_change`) | same | Not stated |
| Element 84 Earth Search (Sentinel-2) | Alternative catalogue (`lulc_change`) | https://earth-search.aws.element84.com/v1 | Not stated |
| NASA Black Marble VIIRS (CMR-STAC LAADS) | Night-time lights (`lulc_change`, needs Earthdata login) | https://cmr.earthdata.nasa.gov/stac/LAADS | Not stated |
| AlphaEarth Foundations embeddings (Source Cooperative) | Optional land-cover features (`pipeline/embeddings.py`) | https://data.source.coop/tge-labs/aef/ | CC BY 4.0 |

## OpenStreetMap services & geocoding

| Source | Used for | Link | Licence / terms |
|---|---|---|---|
| OpenStreetMap via osmnx (Overpass API) | Kallimarmaro / Vas. Olgas network for the agent model | https://overpass-api.de/api/interpreter | ODbL |
| OSM Nominatim | Geocoding sensor-station addresses | https://nominatim.openstreetmap.org | Nominatim usage policy (1 req/s) |

## Basemaps

| Source | Used for | Link | Attribution |
|---|---|---|---|
| Esri World Light Gray (base + reference) | "Light grey" basemap | https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer | Esri, HERE, Garmin, © OpenStreetMap contributors |
| Esri World Imagery (+ boundaries and places) | "Satellite" basemap | https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer | Esri, Maxar, Earthstar Geographics |
| Esri World Topo Map | "Topographic" basemap | https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer | Esri, HERE, Garmin, © OpenStreetMap contributors |
| OpenStreetMap tiles | "Streets" basemap | https://tile.openstreetmap.org | © OpenStreetMap contributors |

## Case-study references (Vas. Olgas)

- Kathimerini: https://www.kathimerini.gr/society/564165649/vasilissis-olgas-anoigei-exi-chronia-meta/
- City of Athens press release: https://www.cityofathens.gr/deltio-typoy/charis-doukas-oi-ergasies-sti-vasilissis-olgas-epitelous-oloklirothikan/
- Anaplasi Dimosion Choron: https://anaplassis.gr/featured_item/ergo-leoforos-vasilissis-olgas/
- Hellas Journal (15–20% forecast): https://hellasjournal.com/2026/04/etoimi-i-vasilissis-olgas-oloklirothike-i-anaplasi-epistrefoun-kai-ta-mesa-mazikis-metaforas-tha-ferei-aposymforisi-eos-20-sto-kykloforiako-videos/
- OpenCouncil, Athens council 23 Sep 2026: https://opencouncil.gr/athens/sep23_2026/subjects/cmuft0jlc0k1mwysp1g2cr8gj

## Notes

1. Licences are stated only for the Attica traffic data, the Apps4Athens logger, the AlphaEarth embeddings and the OSM part of the tree points. Check the others before publishing.
2. The base layers (districts, neighbourhoods, roads, land use, pocket parks, building footprints) have no source note in the repository, and `data/raw/gis/ysos.geojson` is neither documented nor used.
