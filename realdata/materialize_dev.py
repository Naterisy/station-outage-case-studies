"""Materialize only the frozen public NOAA development panel (2021 and 2022)."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import numpy as np
import pandas as pd

BASE = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access"
YEARS = (2021, 2022)
LIMIT = 2_000_000

def source(entries, year, sid):
    if year not in YEARS or not isinstance(sid, str) or not re.fullmatch(r"\d{11}", sid):
        raise ValueError("non-development year or invalid station")
    row = entries[(year, sid)]
    url = f"{BASE}/{year}/{sid}.csv"
    if row["url"] != url or not re.fullmatch(r"[a-f0-9]{64}", row.get("sha256", "")):
        raise ValueError(f"missing frozen authoritative source: {year}/{sid}")
    if not isinstance(row["bytes"], int) or not 0 < row["bytes"] <= LIMIT:
        raise ValueError("invalid frozen source size")
    return url, row

def checked(data, expected):
    if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError("source hash mismatch; NOAA may have revised this file")

def daily(data, sid, year):
    raw = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
    if not raw.STATION.eq(sid).all():
        raise ValueError("station mismatch")
    day = pd.to_datetime(raw.DATE, format="%Y-%m-%d", errors="raise")
    if not day.dt.year.eq(year).all() or day.duplicated().any():
        raise ValueError("dates must be unique and in source year")
    t = pd.to_numeric(raw.TEMP, errors="raise")
    n = pd.to_numeric(raw.TEMP_ATTRIBUTES, errors="raise")
    if not ((n >= 0)&(n <= 24)&(n == np.floor(n))).all() or not np.isfinite(t).all():
        raise ValueError("invalid recorded temperature or observation count")
    x = ((t-32)*5/9).where((t != 9999.9) & (n >= 4))
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31")
    return pd.Series(x.to_numpy(), index=day).reindex(idx).to_numpy(dtype=np.float64)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True, type=Path,
                   help="directory with frozen manifest.json and source_files_dev.json")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cached-raw", type=Path)
    opt = p.parse_args()
    panel = json.loads((opt.metadata/"manifest.json").read_text())["stations"]
    rows = json.loads((opt.metadata/"source_files_dev.json").read_text())
    entries = {(row["year"],row["station"]):row for row in rows}
    if len(entries)!=len(rows) or any(year not in YEARS for year,_ in entries):
        raise ValueError("source inventory must be unique and development-only")
    ids = [r["station"] for r in panel]
    if len(ids) != 36 or len(ids)!=len(set(ids)):
        raise ValueError("frozen panel must contain 36 distinct stations")
    completed = {}
    for year in YEARS:
        values = []
        for sid in ids:
            url, entry = source(entries,year,sid)
            try:
                if opt.cached_raw:
                    data = (opt.cached_raw/str(year)/(sid+".csv")).read_bytes()
                else:
                    with urlopen(url,timeout=35) as r:
                        if r.geturl() != url:
                            raise ValueError("source redirected from immutable verified host/path")
                        data = r.read(LIMIT+1)
                checked(data,entry)
                values.append(daily(data,sid,year))
            except (OSError, ValueError) as e:
                raise SystemExit(f"{year}/{sid}: {e}") from e
        completed[year] = np.stack(values,axis=1)
    valid = [int(np.isfinite(completed[y]).sum()) for y in YEARS]
    if valid!=[13090,13074]:
        raise ValueError(f"unexpected developed panel counts {valid}")
    opt.out.mkdir(parents=True,exist_ok=True)
    for year,x in completed.items():
        np.savez_compressed(opt.out/f"{year}.npz",temperature_c=x,observed=np.isfinite(x),
            dates=pd.date_range(f"{year}-01-01",f"{year}-12-31").to_numpy(dtype="datetime64[D]"),
            station_ids=np.array(ids),coordinates=np.array([[r["lat"],r["lon"]] for r in panel]))
    print(f"Verified 72 snapshots; usable panel values {valid}; wrote 2021 and 2022 only")

if __name__ == "__main__":
    main()
