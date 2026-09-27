"""Checks for analytical assumptions that should survive notebook/module edits."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pandas.testing import assert_frame_equal

from reliability_detector import detect_reliability_events
from reliability_detector.data import build_station_days, load_data
from reliability_detector.detection import (
    detect_systemic_events,
    flag_station_impairment,
)
from reliability_detector.summaries import (
    estimate_energy_shortfall,
    firmware_period_rates,
    summarize_episodes,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.paths = [
            folder / name for name in ("sites.csv", "stations.csv", "attempts.csv")
        ]
        pd.DataFrame({"site_id": ["site"], "type": ["retail"], "stations": [2]}).to_csv(
            self.paths[0], index=False
        )
        pd.DataFrame(
            {
                "station_id": ["active", "quiet"],
                "site_id": ["site", "site"],
                "reader_type": ["reader", "reader"],
                "firmware_version": ["v1", "v1"],
            }
        ).to_csv(self.paths[1], index=False)
        pd.DataFrame(
            {
                "station_id": ["active"] * 5,
                "attempt_timestamp": [
                    "2025-01-06 08:00",
                    "2025-01-06 09:00",
                    "2025-01-06 10:00",
                    "2025-01-06 11:00",
                    "2025-01-20 08:00",
                ],
                "session_start_timestamp": [
                    None,
                    None,
                    None,
                    "2025-01-06 11:00",
                    "2025-01-20 08:00",
                ],
                "session_end_timestamp": [
                    None,
                    None,
                    None,
                    "2025-01-06 11:30",
                    "2025-01-20 08:30",
                ],
                "attempt_outcome": [
                    "auth error",
                    "auth error",
                    "start error",
                    "session",
                    "session",
                ],
                "kwh": [0, 0, 0, 30, 30],
                "duration_min": [0, 0, 0, 30, 30],
            }
        ).to_csv(self.paths[2], index=False)
        self.tables = load_data(*self.paths)
        self.days = build_station_days(*self.tables)

    def test_zero_attempt_days_remain_in_weekday_baseline(self):
        self.assertEqual(len(self.days), 2 * 15)
        mondays = self.days.loc[
            self.days["station_id"].eq("active") & self.days["day_of_week"].eq(1)
        ]
        self.assertEqual(mondays["attempt_count"].tolist(), [4, 0, 1])
        self.assertEqual(mondays["expected_attempt_count"].tolist(), [1, 1, 1])
        self.assertEqual(mondays["demand_pct"].tolist(), [400, 0, 100])

    def test_start_rate_excludes_authorization_errors(self):
        row = self.days.loc[
            self.days["station_id"].eq("active") & self.days["date"].eq("2025-01-06")
        ].iloc[0]
        self.assertEqual(row["auth_rate_pct"], 50)
        self.assertEqual(
            row["start_rate_pct"], 50
        )  # One session / two observable starts.
        self.assertEqual(row["median_power_kw"], 60)

    def test_quiet_station_has_undefined_rates_and_no_low_activity_flag(self):
        before = self.days.copy(deep=True)
        flagged = flag_station_impairment(self.days)
        assert_frame_equal(self.days, before)
        quiet = flagged.loc[flagged["station_id"].eq("quiet")]
        self.assertTrue(
            quiet[["auth_rate_pct", "start_rate_pct", "demand_pct"]].isna().all().all()
        )
        self.assertFalse(quiet["demand_impaired"].any())

    def test_no_event_output_retains_dates_and_null_event_fields(self):
        daily = detect_reliability_events(*self.paths)
        self.assertEqual(len(daily), 15)
        self.assertFalse(daily["has_systemic_failure"].any())
        self.assertTrue(daily["num_stations_systemic"].eq(0).all())
        self.assertTrue(
            daily[
                [
                    "event_type",
                    "station_ids",
                    "correlates",
                    "auth_rate_pct",
                    "start_rate_pct",
                    "demand_pct",
                ]
            ]
            .isna()
            .all()
            .all()
        )
        impact = estimate_energy_shortfall(self.days, daily)
        self.assertTrue(impact.empty)
        self.assertTrue(summarize_episodes(impact).empty)


class SummaryTests(unittest.TestCase):
    def test_shortfall_offsets_surplus_before_clipping_and_splits_date_gaps(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-04"])
        metrics = pd.DataFrame(
            [
                {
                    "date": date,
                    "station_id": station,
                    "expected_kwh_delivered": expected,
                    "total_kwh_delivered": actual,
                }
                for i, date in enumerate(dates)
                for station, expected, actual in [
                    ("A", 100, 100 if i == 1 else 20),
                    ("B", 100, 150),
                    ("unselected", 1000, 0),
                ]
            ]
        )
        daily = pd.DataFrame(
            {
                "date": dates.strftime("%Y-%m-%d"),
                "has_systemic_failure": True,
                "event_type": "site-associated",
                "correlates": "site_id: site",
                "station_ids": [["A", "B"] for _ in dates],
                "num_stations_systemic": 2,
            }
        )
        impact = estimate_energy_shortfall(metrics, daily)
        self.assertEqual(impact["estimated_kwh_shortfall"].tolist(), [30, 0, 30])
        self.assertEqual(impact["episode_id"].tolist(), [1, 1, 2])
        episodes = summarize_episodes(impact)
        self.assertEqual(episodes["duration_days"].tolist(), [2, 1])

    def test_firmware_period_rates_pool_counts_instead_of_averaging_daily_rates(self):
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "firmware_version": ["v1", "v1"],
                "sessions": [10, 0],
                "start_errors": [0, 1],
            }
        )
        rates = firmware_period_rates(
            daily, pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-02")
        )
        self.assertAlmostEqual(rates.loc["v1", "Whole period"], 100 * 10 / 11)
        self.assertEqual(rates.loc["v1", "Outside event"], 100)
        self.assertEqual(rates.loc["v1", "During event"], 0)


class IncludedDatasetTests(unittest.TestCase):
    def test_firmware_cohorts_preserve_timing_and_capture_broader_scope(self):
        data = Path(__file__).resolve().parents[1] / "data"
        tables = load_data(
            *(data / name for name in ("sites.csv", "stations.csv", "attempts.csv"))
        )
        days = flag_station_impairment(build_station_days(*tables))
        combined = detect_systemic_events(days)
        single = detect_systemic_events(days, include_firmware_cohorts=False)
        self.assertEqual(len(combined), 84)
        self.assertEqual(int(combined["has_systemic_failure"].sum()), 24)
        event = combined["date"].between("2025-05-14", "2025-05-18")
        self.assertTrue(combined.loc[event, "has_systemic_failure"].all())
        self.assertTrue(single.loc[event, "has_systemic_failure"].all())
        self.assertEqual(
            combined.loc[event, "num_stations_systemic"].tolist(), [61, 61, 61, 61, 59]
        )
        self.assertEqual(
            single.loc[event, "num_stations_systemic"].tolist(), [37, 37, 37, 37, 36]
        )
        impact = estimate_energy_shortfall(days, combined)
        self.assertAlmostEqual(
            impact["estimated_kwh_shortfall"].sum(), 72276.89, places=2
        )


if __name__ == "__main__":
    unittest.main()
