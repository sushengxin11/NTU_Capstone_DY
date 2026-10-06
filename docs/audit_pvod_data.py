"""Read-only PVOD profiling; write reviewed counts to JSON. Uses Python stdlib only."""
import collections
import csv
import datetime as dt
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "PVODdatasets_v1.0"
OUTPUT = Path(__file__).with_name("PVOD数据核查_2026-10-05.json")


def profile(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames
    times = [dt.datetime.fromisoformat(r["date_time"]) for r in rows]
    unique = sorted(set(times))
    step = dt.timedelta(minutes=15)
    grid = set()
    current = unique[0]
    while current <= unique[-1]:
        grid.add(current)
        current += step
    missing = collections.Counter()
    invalid = collections.Counter()
    values = collections.defaultdict(list)
    for row in rows:
        for col in columns[1:]:
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                missing[col] += 1
                continue
            if not math.isfinite(value):
                missing[col] += 1
                continue
            values[col].append(value)
            if col == "power" and value < 0:
                invalid["negative_power"] += 1
            if "irrad" in col and value < 0:
                invalid["negative_" + col] += 1
            if col == "nwp_humidity" and not 0 <= value <= 100:
                invalid["humidity_outside_0_100"] += 1
            if "windspeed" in col and value < 0:
                invalid["negative_" + col] += 1
            if "winddirection" in col and not 0 <= value <= 360:
                invalid[col + "_outside_0_360"] += 1
    counter = collections.Counter(times)
    hours = collections.defaultdict(list)
    for time, row in zip(times, rows):
        hours[time.hour].append(float(row["power"]))
    return {
        "station": path.stem,
        "rows": len(rows),
        "columns": columns,
        "start_utc": str(unique[0]),
        "end_utc": str(unique[-1]),
        "expected_grid_rows": len(grid),
        "missing_grid_timestamps": len(grid - set(times)),
        "off_grid_timestamps": len(set(times) - grid),
        "duplicate_extra_rows": len(times) - len(unique),
        "duplicate_timestamps": {str(t): n for t, n in counter.items() if n > 1},
        "backward_steps": sum(b < a for a, b in zip(times, times[1:])),
        "missing_or_nonfinite_fields": dict(missing),
        "basic_domain_flags": dict(invalid),
        "power_min": min(values["power"]),
        "power_max": max(values["power"]),
        "mean_power_peak_utc_hour": max(hours, key=lambda h: sum(hours[h]) / len(hours[h])),
        "april_2019_utc_rows": sum(t.year == 2019 and t.month == 4 for t in times),
    }


if __name__ == "__main__":
    with (DATA / "metadata.csv").open(encoding="utf-8-sig", newline="") as f:
        metadata = list(csv.DictReader(f))
    stations = [profile(p) for p in sorted(DATA.glob("station*.csv"))]
    result = {
        "scope": "All station CSV rows; timestamps interpreted as UTC per author toolkit. No raw data edits.",
        "metadata_station_count": len(metadata),
        "metadata_station_ids": [r["Station_ID"] for r in metadata],
        "total_station_rows": sum(r["rows"] for r in stations),
        "stations": stations,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("total rows:", result["total_station_rows"])
    for r in stations:
        print(r["station"], r["rows"], "missing grid:", r["missing_grid_timestamps"], "duplicates:", r["duplicate_extra_rows"])
    print("output:", OUTPUT)
