"""One-off preparation of everything the demo shows. Run from the project root:
    python -m urbanintel.prepare            # layers only (fast, offline)
    python -m urbanintel.prepare --cases    # also run and cache the case-study simulations (needs OSM access the first time)
    python -m urbanintel.prepare --force    # rebuild layers that already exist"""
import argparse

from .services import overview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cases", action="store_true")
    a = ap.parse_args()
    print("Preparing layers ->", overview.LY.LAYER_DIR)
    overview.prepare_all(force=a.force)
    if a.cases:
        from .services import flood, heat, mobility
        print("Flood exposure ...")
        flood.city_exposure(force=True)
        flood.network_impact(force=True)
        try:
            print("Sentinel-2 wet-season indices (network) ...")
            flood.spectral(force=False)
        except Exception as e:      # offline: the rest of the demo still works
            print(f"  skipped: {e}")
        print("Vas. Olgas mobility ABM ...")
        mobility.run_case(force=True)
        try:
            from .services import landcover
            print("Vas. Olgas Sentinel-2 land-cover change (network) ...")
            landcover.run(force=False)
        except Exception as e:
            print(f"  skipped: {e}")
        print("Heat ABM ...")
        heat.run_case(force=True)
    print("done")


if __name__ == "__main__":
    main()
