"""Import weekday-peak Attica traffic readings from the Apps4Athens data-logger's GitHub Actions artifacts.

The Region of Attica feed keeps no history; a hackathon project (github.com/nickoulos/apps4athens-data-logger, MIT code, CC BY 4.0
data) logs it every minute since 21 Oct 2026 and uploads ~70-minute chunks as Actions artifacts (kept 90 days). Downloading an
artifact needs any GitHub token (public repo, no scopes). The token is read from $GITHUB_TOKEN or ~/.github_token, never printed.

Only artifacts overlapping weekday peaks (07-10 and 17-20 Athens time) are fetched, and only detectors within --radius m of
Vas. Olgas are kept. Rows are appended to data/raw/traffic/attica/YYYY-MM-DD.csv.gz in the platform's own logger format, so
`python -m urbanintel.calibrate` uses them for the weekday-peak ('after') calibration.

    python -m urbanintel.fetch_logger_artifacts                # weekday peaks, detectors within 1500 m of Olgas
    python -m urbanintel.fetch_logger_artifacts --all-hours    # every hour (much larger download)"""
import argparse
import gzip
import io
import json
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .services import traffic

REPO = "nickoulos/apps4athens-data-logger"
API = f"https://api.github.com/repos/{REPO}/actions/artifacts"
CACHE = traffic.LY.ROOT / "data" / "cache" / "gh_artifacts"
PEAK_HOURS = set(range(7, 10)) | set(range(17, 20))


def token():
    t = os.environ.get("GITHUB_TOKEN")
    if not t:
        p = Path.home() / ".github_token"
        t = p.read_text().strip() if p.exists() else None
    if not t:
        raise SystemExit("No token: set GITHUB_TOKEN or write it to ~/.github_token (chmod 600)")
    return t


def _req(url, tok):
    return urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
                                                "User-Agent": "athens-urban-intelligence"})


def list_artifacts(tok):
    out, page = [], 1
    while True:
        with urllib.request.urlopen(_req(f"{API}?per_page=100&page={page}", tok), timeout=60) as r:
            d = json.load(r)
        out += [a for a in d["artifacts"] if not a["expired"] and a["name"].startswith("data-")]
        if len(d["artifacts"]) < 100:
            return out
        page += 1


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kw):
        return None


def download(artifact_id, tok):
    """GitHub answers with a 302 to a signed storage URL; follow it WITHOUT the GitHub token (the storage rejects it)."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(_req(f"{API}/{artifact_id}/zip", tok), timeout=60)
        raise RuntimeError("expected a redirect")
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        loc = e.headers["Location"]
    with urllib.request.urlopen(urllib.request.Request(loc, headers={"User-Agent": "athens-urban-intelligence"}), timeout=180) as r:
        return r.read()


def wanted(a, all_hours=False):
    """True if the artifact's ~70-minute window touches a weekday peak hour in Athens time."""
    end = pd.Timestamp(a["created_at"]).tz_convert("Europe/Athens")
    for t in (end, end - pd.Timedelta(minutes=35), end - pd.Timedelta(minutes=70)):
        if t.dayofweek < 5 and (all_hours or t.hour in PEAK_HOURS):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=1500)
    ap.add_argument("--all-hours", action="store_true")
    a = ap.parse_args()
    tok = token()
    keep = set(traffic.near_olgas(a.radius).id)
    arts = [x for x in list_artifacts(tok) if wanted(x, a.all_hours)]
    done_f = CACHE / "imported.json"
    CACHE.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
    todo = [x for x in arts if str(x["id"]) not in done]
    print(f"{len(arts)} artifacts overlap weekday peaks; {len(todo)} new; keeping {len(keep)} detectors within {a.radius:.0f} m of Olgas")
    traffic.LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = 0
    for i, x in enumerate(todo, 1):
        z = zipfile.ZipFile(io.BytesIO(download(x["id"], tok)))
        for name in z.namelist():
            if not (name.startswith("traffic/") and name.endswith(".csv.gz")):
                continue
            df = pd.read_csv(io.BytesIO(gzip.decompress(z.read(name))))
            df = df[(df.vehicle == "all") & df.id.isin(keep)][["id", "timestamp", "count", "occ", "speed"]]
            if df.empty:
                continue
            for day, g in df.groupby(df.timestamp.str[:10]):
                p = traffic.LOG_DIR / f"{day}.csv.gz"
                head = not p.exists()
                with gzip.open(p, "at") as f:
                    g.to_csv(f, header=head, index=False)
            rows += len(df)
        done.add(str(x["id"]))
        if i % 10 == 0 or i == len(todo):
            done_f.write_text(json.dumps(sorted(done)))
            print(f"  {i}/{len(todo)} artifacts, {rows:,} rows kept", flush=True)
    print("done")


if __name__ == "__main__":
    main()
