# Yoken: urban intelligence demo (Athens)

Self-contained copy of the demo app (Solara UI + FastAPI) with the data it needs.

## Install
```bash
conda create -n yoken -c conda-forge python=3.11 geopandas rasterio osmnx
conda activate yoken
pip install -r requirements.txt -r requirements-demo.txt
```

## Run (from this folder)
```bash
solara run urbanintel/ui/app.py --port 8765        # UI   -> http://localhost:8765
uvicorn urbanintel.api.main:app --port 8000        # API  -> http://localhost:8000/docs
```

## What is included
- `urbanintel/`: UI, API, services, case-study config (`urbanintel/config/cases.yaml`).
- `pipeline/`: agent model, Sentinel-2 / Landsat fetch, spectral indices.
- `data/outputs/demo/layers`: the prepared map layers (all 29 overview layers).
- `data/outputs/cases`: cached results for Vas. Olgas, Heat and Flood (the demo loads instantly).
- `data/outputs/timelapse`: built timelapses.
- `data/raw/traffic`, `data/raw/environment`: Attica traffic detectors, City of Athens sensor stations.
- `gis_Data/` (6 files), `data/raw/gis/` (4 files) and `data/cache/osm`: inputs for new heat / traffic simulations and the district selector.
- `DATA_SOURCES.md`: all data sources with links. `docs/yoken_demo.gif|mp4`: demo recording.

## Needs the internet
- New Sentinel-2 periods (Flood tab) and new timelapses: Microsoft Planetary Computer.
- Live traffic feed: data.gov.gr.
- Basemaps: Esri / OpenStreetMap tiles.

## Not included (only needed to rebuild from scratch)
The rest of `gis_Data/` and `data/raw/gis/` (~700 MB), satellite scene caches and the Streamlit
analyst app. `python -m urbanintel.prepare` and `python -m urbanintel.calibrate` need those, so run them
in the original project and copy `data/outputs/` across.
