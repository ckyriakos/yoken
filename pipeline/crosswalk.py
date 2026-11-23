"""Match sensor-station neighbourhood labels to layer polygons (accent-, case- and alphabet-insensitive)."""
import difflib, re, unicodedata
import pandas as pd

LATIN2GREEK = str.maketrans("ABEHIKMNOPTXYZ", "ΑΒΕΗΙΚΜΝΟΡΤΧΥΖ")   # look-alike Latin capitals -> Greek


def norm(s):
    s = unicodedata.normalize("NFD", str(s).upper())
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).translate(LATIN2GREEK)
    s = s.replace("ς", "Σ")
    return re.sub(r"\s+", " ", s).strip()


def match_stations(stations, layer, name_col="name", id_col="id", dk_col="dk"):
    """stations: df[station_id, station_name, neighbourhood_name, municipal_district_id].
    layer: df[name, id, dk]. Returns one row per station with method and a review flag."""
    L = layer.assign(_n=layer[name_col].map(norm), _d=layer[dk_col].astype(str).str.extract(r"(\d)")[0].astype(float))
    rows = []
    for _, s in stations.drop_duplicates("station_id").iterrows():
        n, method = norm(s.neighbourhood_name), None
        cand = L[L._n == n]
        if len(cand):
            method = "exact"
        elif "-" in n:
            parts = [p.strip() for p in n.split("-")]
            c2 = L[L._n.isin(parts)]
            if len(c2): cand, method = c2, "compound_part"
        if method is None:
            close = difflib.get_close_matches(n, L._n.unique(), n=1, cutoff=0.85)
            if close: cand, method = L[L._n == close[0]], "fuzzy"
        if method and "municipal_district_id" in s and pd.notna(s.municipal_district_id) and len(cand) > 1:
            same = cand[cand._d == s.municipal_district_id]
            if len(same) == 1: cand = same
        row = dict(station_id=s.station_id, station_name=s.station_name, station_nbhd=s.neighbourhood_name,
                   station_district=s.get("municipal_district_id"), method=method or "unmatched",
                   layer_id=None, layer_name=None, layer_district=None, review=True)
        if method and len(cand) == 1:
            r = cand.iloc[0]
            row.update(layer_id=r[id_col], layer_name=r[name_col], layer_district=r._d)
            d_ok = pd.isna(row["station_district"]) or r._d == row["station_district"]
            row["review"] = method in ("fuzzy", "compound_part") or not d_ok
            row["district_mismatch"] = not d_ok
        elif method:
            row["method"] = method + "_ambiguous"
        rows.append(row)
    return pd.DataFrame(rows)
