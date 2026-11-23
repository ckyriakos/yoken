"""Log the Region of Attica live traffic feed (it keeps no history). Run from the project root:
    python -m urbanintel.collect_traffic                 # poll every 90 s until stopped
    python -m urbanintel.collect_traffic --minutes 60    # poll for an hour
Writes data/raw/traffic/attica/YYYY-MM-DD.csv.gz (UTC days). One request per poll; the API allows 60 per minute."""
import argparse
import time

from .services import traffic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0, help="0 = run until stopped")
    ap.add_argument("--every", type=float, default=90, help="seconds between polls")
    a = ap.parse_args()
    traffic.stations(refresh=True)
    end = time.time() + a.minutes * 60 if a.minutes else float("inf")
    state = {}
    while time.time() < end:
        try:
            n = traffic.collect(state)
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"+{n} readings", flush=True)
        except Exception as e:      # network hiccup: keep going
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "error:", e, flush=True)
        time.sleep(a.every)


if __name__ == "__main__":
    main()
