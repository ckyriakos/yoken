# Athens Urban Intelligence: demo UI + API

The demo has a simple 4-page UI (Solara + leafmap) and an HTTP API (FastAPI). Both run on the same service layer in `urbanintel/services/`. The old Streamlit `app.py` is kept untouched as the analyst view.

## Run

```bash
pip install -r requirements.txt -r requirements-demo.txt
python -m urbanintel.prepare --cases      # once: web-ready layers + cached case results (OSM + Sentinel-2 need network the first time)
python -m urbanintel.collect_traffic      # keep running: logs the Attica loop-detector feed (it keeps no history)
python -m urbanintel.calibrate            # fit car demand to the detectors and re-run the Vas. Olgas case
solara run urbanintel/ui/app.py           # UI  -> http://localhost:8765
uvicorn urbanintel.api.main:app --port 8000   # API -> http://localhost:8000/docs
```

## Pages

| Page | What it shows | Service |
|---|---|---|
| Overview | All 29 layers of `gis_Data/` plus heat outputs, toggled by group; 4 headline numbers | `services/overview.py` |
| Case: Vas. Olgas | The 2026 redevelopment as built (open to cars both ways, ~150 trees, new end signals, Amalias U-turn) against the closed street, plus traffic-only and trees-only variants. Car agents use time-stepped BPR assignment; resident agents walk with sun aversion and crossing waits created by the cars. Flow on Amalias / Vas. Konstantinou / Arditou is compared with the municipality's 15–20% decongestion forecast. Sentinel-2 NDVI for summer 2019 (before), 2025 (works) and 2026 (after), with a control ring | `services/mobility.py`, `services/landcover.py` |
| Case: Heat | Street trees and cooling refuges proposed for the top-3 heat-priority neighbourhoods, then simulated. Each proposal changes shade **and** surface temperature on the same streets. Includes a 65+ subgroup | `services/heat.py` |
| Case: Flood | Residents in the 50/100/1000-yr zones, walking access lost when flooded streets close, and wet-season Sentinel-2 NDWI/MNDWI/NDBI/NDVI with a runoff-susceptibility screen | `services/flood.py` |

**Sensors** (Overview, own group): the City of Athens environmental stations (hourly air temperature, humidity, AQI, PM, NO₂, O₃; geocoded once from the station registry, 4 placed at their neighbourhood centre) coloured by summer-afternoon air temperature. Clicking a station shows the latest reading, a 7-day temperature line and the neighbourhood's satellite surface temperature. The Region of Attica traffic detectors are in the same group.

**Area types and measures**: the "Area type & fitting measures" layer (Heat group) classifies every neighbourhood (dense residential, mixed use, civic…) and lists the cooling measures that fit it, each marked ● simulated on the Heat page, ◐ approximated or ○ needs data first (from `actions.yaml` and `decision_map.yaml`). The Heat page shows the same for the simulated neighbourhoods under "Why these neighbourhoods, and which measures fit".

**Index cards**: every raster or coloured layer gets the same card: what it is, a colour bar with plain-language ends ("dry, built" ↔ "standing water"), how to read it, the exact image dates, and the formula and source under "More".

**District selector** (top right): pick one or more of the 7 municipal districts. Maps zoom to them and dim the rest. Overview numbers, flood exposure and tables are recomputed for the selection. The Heat case simulates the top-3 heat-priority neighbourhoods inside it, and timelapses are built for it. The Vas. Olgas case stays fixed at Kallimarmaro.

**Timelapse** page: choose Sentinel-2 (2017+, 10–20 m) or Landsat 5/7/8/9 (1984+, 30 m, includes surface temperature), an index (true colour, NDVI, NDWI, MNDWI, NDBI, surface temperature), a year range, a season and yearly or monthly steps. For each step the platform downloads the clearest scenes from the Planetary Computer, masks clouds, takes the median and computes the index. It then renders a titled frame (animated GIF, downloadable), a map overlay (frame slider) and the area mean (trend chart). Results are cached in `data/outputs/timelapse/<key>/`; scene bands in `data/cache/`.

The right-hand "Ask Athens" box on the Overview page is a placeholder for the planned chat. The chat will call the API endpoints below as tools.

## API (`/api/v1`)

- `GET /layers` (catalogue), `GET /layers/{id}?bbox=minLon,minLat,maxLon,maxLat` (GeoJSON, or raster metadata), `GET /layers/{id}/image.png`, `GET /layers/{id}/tiles/{z}/{x}/{y}.png`
- `GET /indicators/overview`, `GET /indicators/neighbourhoods?format=json|geojson`
- `GET /cases`, `GET /cases/{kallimarmaro|heat|flood}`, `GET /cases/kallimarmaro/layers/{edges|traces|olgas|signals}`, `GET /cases/kallimarmaro/landcover`
- `POST /simulate/heat` `{"tree_budget_m": 1000, "refuges": 2, "neighbourhoods": [...]}`: cached by parameters, about 3 s when new
- `POST /simulate/kallimarmaro` `{"force": false}`
- `GET /traffic/live?bbox=...` (Attica detectors, cached 60 s), `GET /traffic/history?ids=MS358,MS231&start=&end=` (from the logger), `GET /cases/kallimarmaro/calibration`
- `GET /sensors/environment` (stations + latest values), `GET /sensors/environment/{id}/series?days=14`, `GET /neighbourhoods/{id}/measures`
- `GET /districts`; `?districts=2,4` on `/indicators/overview` and `/indicators/neighbourhoods`
- `POST /timelapse` `{"sensor": "landsat", "product": "lst", "start_year": 2018, "end_year": 2026, "season": "Summer (Jun–Sep)", "districts": [4]}` → background build; `GET /timelapse/{key}` (status, frames with dates and area mean), `GET /timelapse/{key}/timelapse.gif`, `GET /timelapse` (saved)
- `GET /flood/exposure?bbox=...`, `GET /flood/network`
- `POST /indices` `{}` → the local wet-season set (default). `POST /indices` `{"start": "2024-06-01", "end": "2024-08-31", "max_cloud_pct": 20, "max_scenes": 6}` → downloads Sentinel-2 L2A scenes for that period from the Planetary Computer STAC in the background and computes NDWI, MNDWI, NDVI, NDBI and runoff susceptibility; poll `GET /indices/periods/{key}`. `GET /indices/periods` lists all periods. New layers appear as `s2_<index>__<YYYYMMDD_YYYYMMDD>`.

## What "summer" and "wet season" mean

- **Summer 2026** (heat layers): median of 7 cloud-free Landsat 8/9 images, 29 Jun – 1 Sep 2026, taken around midday; NDVI from 7 matching Sentinel-2 images (28 Jun – 1 Sep 2026).
- **Wet season 2026** (water and built-up indices, the default "local" set): median of 4 cloud-free Sentinel-2 images, 3 Feb – 29 Apr 2026, when soils are wettest. Any other period can be computed from the Flood page ("New period…") or the API; results are cached in `data/outputs/cases/flood/runs/`.
- **Vas. Olgas NDVI**: June–September medians of 6 Sentinel-2 images each, for 2019, 2025 and 2026.

The exact dates are read from the data and shown in the map's info box and layer explanations (`urbanintel/explain.py`).

## Assumptions (all in `urbanintel/config/cases.yaml`)

- **Vas. Olgas is the built project** (handed back April 2026, all-vehicle pilot since May 2026; sources in `cases.yaml`). OSM still tags it `living_street`, `motor_vehicle=no`, so the model uses that as "before". "As built" uses 1 lane per direction at 30 km/h, 700 veh/h per lane, 30 bus/tram per hour per direction, new signals at both ends and the Amalias U-turn. No measured pilot results are published; the 15–20% figure is the municipality's forecast.
- **Trees:** 150 young trees remove 25% of the remaining sun on Olgas (assumption; grows as they mature). The 100+ new luminaires are not modelled (daytime model).
- **Land-cover change:** June–September median NDVI, 6 scenes per year, 10 m. Corridor = Olgas ±25 m (3.7 ha); control = 60–400 m ring. Classes use the lulc.yaml thresholds (tree ≥ 0.35, grass ≥ 0.15).
- **Car demand** is synthetic in structure (through trips, local trips, and the Pangrati/Kallimarmaro → Syntagma/Plaka movement) but its level is **calibrated to Region of Attica loop detectors** ([data.gov.gr](https://data.gov.gr/dataset/attica-region-traffic-data), CC BY 4.0). Main-road detectors within 800 m of Olgas are used; bus lanes, ramps and turn lanes are excluded. The first fit uses the only published history, Sat 22 Nov 2025 15:00 (Olgas closed): demand × 1.76, median observed/model 1.02, error per detector 38%. The fit is a weekend one; `python -m urbanintel.calibrate` switches to weekday peaks once the logger has collected them. The history's `countedcars` is read as the sum of 40 × 90 s samples (veh/h = value / 40), inferred from matching live readings.
- **Heat dose:** minutes walked × exposure, where exposure = ½ sun share + ½ normalised LST. The 8-minute threshold and the 23% share of residents aged 65+ are assumptions.
- **Runoff susceptibility** is a screening index, not a hydraulic model.
