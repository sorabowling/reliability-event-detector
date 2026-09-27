"""Generate a fictional DC charging network using only NumPy and pandas.

Run from any directory:
    python generate_data.py --output-dir data --seed 20260927
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2025-04-07")
DAYS = 84
LOCATIONS = [
    ("Portland", "OR", "West"),
    ("Sacramento", "CA", "West"),
    ("Spokane", "WA", "West"),
    ("Boise", "ID", "West"),
    ("Reno", "NV", "West"),
    ("Salt Lake City", "UT", "West"),
    ("Raleigh", "NC", "East"),
    ("Richmond", "VA", "East"),
    ("Pittsburgh", "PA", "East"),
    ("Charlotte", "NC", "East"),
    ("Tampa", "FL", "East"),
    ("Jacksonville", "FL", "East"),
    ("Madison", "WI", "Central"),
    ("Milwaukee", "WI", "Central"),
    ("Omaha", "NE", "Central"),
    ("Tulsa", "OK", "Central"),
    ("Albuquerque", "NM", "West"),
    ("Kansas City", "MO", "Central"),
]
SITE_TYPES = [
    "Retail", "Highway Rest Stop", "Parking Garage",
    "Workplace", "Hotel", "Grocery",
]

# Scenarios are simulator inputs, never columns passed to the detector.
# End dates are inclusive. Exact detection is deliberately not guaranteed.
SCENARIOS = [
    {"mode": "authorization", "feature": "reader_type", "values": ["reader_b"],
     "start": "2025-04-20", "end": "2025-04-28"},
    {"mode": "offline", "feature": "site_id", "values": ["HUB_206"],
     "start": "2025-05-10", "end": "2025-05-12"},
    {"mode": "start", "feature": "firmware_version", "values": ["v4.2.0", "v4.3.0"],
     "start": "2025-05-14", "end": "2025-05-18"},
    {"mode": "low_power", "feature": "type", "values": ["Highway Rest Stop"],
     "start": "2025-05-29", "end": "2025-06-02"},
    {"mode": "reduced_activity", "feature": "site_id", "values": ["HUB_213"],
     "start": "2025-06-11", "end": "2025-06-12"},
    {"mode": "zero_energy", "feature": "station_id", "values": ["CP_1007"],
     "start": "2025-06-18", "end": "2025-06-21"},
]


def generate(output_dir, seed=20260927):
    """Write sites.csv, stations.csv, and attempts.csv; return row counts."""
    rng = np.random.default_rng(seed)
    sites = []
    stations = []
    site_types = rng.permutation(SITE_TYPES * 3)
    station_counts = rng.permutation([4, 6, 8] * 6)
    for i, (city, state, region) in enumerate(LOCATIONS):
        site_id = f"HUB_{201 + i}"
        sites.append([site_id, city, state, region, site_types[i], station_counts[i]])
        for _ in range(station_counts[i]):
            stations.append([
                f"CP_{1001 + len(stations)}", site_id,
                rng.choice(["reader_a", "reader_b", "reader_c"], p=[0.40, 0.35, 0.25]),
                rng.choice(["v4.1.0", "v4.2.0", "v4.3.0"], p=[0.45, 0.30, 0.25]),
            ])

    sites = pd.DataFrame(sites, columns=["site_id", "city", "state", "region", "type", "stations"])
    stations = pd.DataFrame(stations, columns=["station_id", "site_id", "reader_type", "firmware_version"])
    metadata = stations.merge(sites[["site_id", "type"]], on="site_id", validate="many_to_one")
    site_activity = dict(zip(sites.site_id, rng.uniform(0.85, 1.20, len(sites))))
    # A small shared daily fluctuation plus heterogeneous station utilization.
    day_activity = rng.lognormal(-0.5 * 0.07**2, 0.07, DAYS)
    rows = []
    for station in metadata.to_dict("records"):
        base_arrivals = rng.uniform(10, 17) * site_activity[station["site_id"]]
        auth_success = rng.uniform(0.955, 0.985)
        start_success = rng.uniform(0.925, 0.975)
        typical_power_kw = rng.uniform(55, 105)
        available_at = START
        for day_index, day in enumerate(pd.date_range(START, periods=DAYS)):
            modes = {
                scenario["mode"] for scenario in SCENARIOS
                if scenario["start"] <= day.strftime("%Y-%m-%d") <= scenario["end"]
                and station[scenario["feature"]] in scenario["values"]
            }
            if "offline" in modes:
                continue  # Missing logs cannot establish power versus communications loss.
            weekend = day.dayofweek >= 5
            weekend_factor = (0.48 if station["type"] == "Workplace" else 1.22) if weekend else 1.0
            activity_factor = 0.06 if "reduced_activity" in modes else 1.0
            arrival_count = rng.poisson(base_arrivals * weekend_factor * day_activity[day_index] * activity_factor)
            peak_hour = 11.5 if station["type"] == "Workplace" else 14.0
            arrival_hours = np.where(
                rng.random(arrival_count) < 0.18,
                rng.uniform(0, 24, arrival_count),
                np.clip(rng.normal(peak_hour, 3.5, arrival_count), 0.02, 23.90),
            )
            # One station represents one connector. Busy arrivals leave unrecorded.
            for arrival_hour in sorted(arrival_hours):
                attempted_at = day + pd.Timedelta(seconds=int(arrival_hour * 3600))
                if attempted_at < available_at:
                    continue
                for retry in range(3):
                    p_auth = 0.32 if "authorization" in modes else auth_success
                    p_start = 0.28 if "start" in modes else start_success
                    if rng.random() > p_auth:
                        outcome = "auth error"
                    elif rng.random() > p_start:
                        outcome = "start error"
                    else:
                        outcome = "session"

                    if outcome == "session":
                        started_at = attempted_at + pd.Timedelta(seconds=int(rng.integers(8, 50)))
                        zero_probability = 0.70 if "zero_energy" in modes else 0.003
                        if rng.random() < zero_probability:
                            kwh = 0.0
                            duration_min = round(float(rng.uniform(1.0, 4.0)), 1)
                        else:
                            kwh = round(float(np.clip(rng.lognormal(np.log(25), 0.40), 6, 72)), 2)
                            power_kw = rng.uniform(10, 19) if "low_power" in modes else np.clip(
                                rng.normal(typical_power_kw, 12), 32, 130
                            )
                            duration_min = round(float(60 * kwh / power_kw), 1)
                        ended_at = started_at + pd.Timedelta(seconds=round(duration_min * 60))
                        rows.append([station["station_id"], attempted_at, outcome, started_at,
                                     ended_at, kwh, duration_min])
                        available_at = ended_at + pd.Timedelta(seconds=60)
                        break

                    rows.append([station["station_id"], attempted_at, outcome, pd.NaT, pd.NaT, np.nan, np.nan])
                    available_at = attempted_at + pd.Timedelta(seconds=30)
                    # Retries increase observed attempts during failures without more visitors.
                    if retry == 2 or rng.random() >= 0.60:
                        break
                    next_attempt = attempted_at + pd.Timedelta(seconds=int(rng.integers(35, 110)))
                    if next_attempt.normalize() != day:
                        break
                    attempted_at = next_attempt

    attempts = pd.DataFrame(rows, columns=[
        "station_id", "attempt_timestamp", "attempt_outcome", "session_start_timestamp",
        "session_end_timestamp", "kwh", "duration_min",
    ]).sort_values(["attempt_timestamp", "station_id"], ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("sites", sites), ("stations", stations), ("attempts", attempts)]:
        frame.to_csv(output_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    return {"sites": len(sites), "stations": len(stations), "attempts": len(attempts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    print(generate(args.output_dir, args.seed))
