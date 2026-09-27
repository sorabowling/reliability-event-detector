"""Flag station impairment and select the strongest shared-group event per day.

Station tests are unadjusted at 1%. Group tests use a separate daily Bonferroni
correction. Selected associations support investigation, not causal attribution.
"""

from itertools import combinations

import pandas as pd
from scipy.stats import binom, fisher_exact, poisson

from .data import build_station_days, load_data

IMPAIRMENT_COLUMNS = [
    "auth_impaired",
    "start_impaired",
    "demand_impaired",
    "power_impaired",
]
CANDIDATE_KEYS = [
    "date",
    "event_type",
    "correlate_feature",
    "correlate_value",
    "failure_mode",
]


def flag_station_impairment(station_day_df):
    """Return a copy with authorization, start, activity, power, and overall flags.

    Uses median daily pooled-network success rates as binomial references.
    Zero expected activity skips the Poisson test. Power rules require four
    sessions below 25 kW median, or two zero-energy sessions.
    """
    station_day_df = station_day_df.copy()
    observable_start_count = (
        station_day_df["session_count"] + station_day_df["start_error_count"]
    )
    daily_auth_rate = (
        1
        - station_day_df.groupby("date")["auth_error_count"].sum()
        / station_day_df.groupby("date")["attempt_count"].sum()
    )
    normal_auth_rate = daily_auth_rate.median()

    daily_start_counts = station_day_df.groupby("date")[
        [
            "session_count",
            "start_error_count",
        ]
    ].sum()
    daily_start_rate = daily_start_counts["session_count"] / (
        daily_start_counts["session_count"] + daily_start_counts["start_error_count"]
    )
    normal_start_rate = daily_start_rate.median()

    station_day_df["auth_impaired"] = station_day_df["attempt_count"].gt(0) & (
        binom.cdf(
            station_day_df["attempt_count"] - station_day_df["auth_error_count"],
            station_day_df["attempt_count"],
            normal_auth_rate,
        )
        < 0.01
    )
    station_day_df["start_impaired"] = observable_start_count.gt(0) & (
        binom.cdf(
            station_day_df["session_count"],
            observable_start_count,
            normal_start_rate,
        )
        < 0.01
    )
    station_day_df["demand_impaired"] = station_day_df["expected_attempt_count"].gt(
        0
    ) & (
        poisson.cdf(
            station_day_df["attempt_count"],
            station_day_df["expected_attempt_count"],
        )
        < 0.01
    )
    station_day_df["power_impaired"] = (
        station_day_df["session_count"].ge(4) & station_day_df["median_power_kw"].lt(25)
    ) | station_day_df["zero_kwh_session_count"].ge(2)

    station_day_df["is_impaired"] = station_day_df[IMPAIRMENT_COLUMNS].any(axis=1)

    return station_day_df


def _build_group_membership(station_day_df, *, include_firmware_cohorts):
    """Expand station-days into site, reader, firmware, and site-type groups.

    Single-category groups use any impairment. Multi-version firmware
    cohorts use start impairment, with at least one version outside the cohort.
    """
    membership_columns = [
        "date",
        "station_id",
        "is_impaired",
        "auth_rate_pct",
        "start_rate_pct",
        "demand_pct",
        "event_type",
        "correlate_feature",
        "correlate_value",
        "failure_mode",
    ]
    firmware_membership_frames = []
    firmware_versions = sorted(
        station_day_df["firmware_version"].dropna().unique().tolist()
    )
    cohort_sizes = range(2, len(firmware_versions)) if include_firmware_cohorts else []
    for cohort_size in cohort_sizes:
        for firmware_cohort in combinations(
            firmware_versions,
            cohort_size,
        ):
            cohort_mask = station_day_df["firmware_version"].isin(firmware_cohort)
            firmware_membership_frames.append(
                station_day_df.loc[cohort_mask].assign(
                    is_impaired=station_day_df.loc[
                        cohort_mask,
                        "start_impaired",
                    ],
                    event_type="firmware-associated",
                    correlate_feature="firmware_version",
                    correlate_value=" + ".join(firmware_cohort),
                    failure_mode="start",
                )
            )

    system_membership_df = pd.concat(
        [
            station_day_df.assign(
                event_type="site-associated",
                correlate_feature="site_id",
                correlate_value=station_day_df["site_id"],
                failure_mode="any",
            ),
            station_day_df.assign(
                event_type="firmware-associated",
                correlate_feature="firmware_version",
                correlate_value=station_day_df["firmware_version"],
                failure_mode="any",
            ),
            station_day_df.assign(
                event_type="reader-associated",
                correlate_feature="reader_type",
                correlate_value=station_day_df["reader_type"],
                failure_mode="any",
            ),
            station_day_df.assign(
                event_type="site-type-associated",
                correlate_feature="site_type",
                correlate_value=station_day_df["site_type"],
                failure_mode="any",
            ),
            *firmware_membership_frames,
        ],
        ignore_index=True,
    )[membership_columns]

    return system_membership_df


def _score_groups(station_day_df, system_membership_df):
    """Compare impairment inside and outside each group using one-sided Fisher tests."""
    systemic_candidates_df = system_membership_df.groupby(
        CANDIDATE_KEYS,
        as_index=False,
    ).agg(
        total_stations=("station_id", "nunique"),
        num_stations_impaired=("is_impaired", "sum"),
    )
    systemic_candidates_df["impaired_pct"] = (
        100
        * systemic_candidates_df["num_stations_impaired"]
        / systemic_candidates_df["total_stations"]
    )

    daily_impaired = station_day_df.groupby("date")["is_impaired"].sum()
    network_station_count = station_day_df["station_id"].nunique()
    systemic_candidates_df["network_stations_impaired"] = systemic_candidates_df[
        "date"
    ].map(daily_impaired)
    for impairment_column in IMPAIRMENT_COLUMNS:
        failure_mode = impairment_column.replace(
            "_impaired",
            "",
        )
        mode_daily_impaired = station_day_df.groupby("date")[impairment_column].sum()
        mode_mask = systemic_candidates_df["failure_mode"].eq(failure_mode)
        systemic_candidates_df.loc[
            mode_mask,
            "network_stations_impaired",
        ] = systemic_candidates_df.loc[
            mode_mask,
            "date",
        ].map(mode_daily_impaired)
    systemic_candidates_df["impaired_outside_group"] = (
        systemic_candidates_df["network_stations_impaired"]
        - systemic_candidates_df["num_stations_impaired"]
    )
    systemic_candidates_df["healthy_outside_group"] = (
        network_station_count
        - systemic_candidates_df["total_stations"]
        - systemic_candidates_df["impaired_outside_group"]
    )

    systemic_candidates_df["p_value"] = systemic_candidates_df.apply(
        lambda row: (
            fisher_exact(
                [
                    [
                        row["num_stations_impaired"],
                        row["total_stations"] - row["num_stations_impaired"],
                    ],
                    [
                        row["impaired_outside_group"],
                        row["healthy_outside_group"],
                    ],
                ],
                alternative="greater",
            ).pvalue
        ),
        axis=1,
    )
    tests_per_date = systemic_candidates_df.groupby("date")[
        "correlate_value"
    ].transform("size")
    systemic_candidates_df["significance_threshold"] = 0.01 / tests_per_date
    systemic_candidates_df["is_systemic"] = systemic_candidates_df[
        "num_stations_impaired"
    ].ge(2) & systemic_candidates_df["p_value"].lt(
        systemic_candidates_df["significance_threshold"]
    )

    return systemic_candidates_df


def _select_daily_events(station_day_df, system_membership_df, systemic_candidates_df):
    """Select by p-value, then impaired fraction, then count; retain every date."""
    systemic_events_df = (
        systemic_candidates_df.loc[systemic_candidates_df["is_systemic"]]
        .sort_values(
            [
                "date",
                "p_value",
                "impaired_pct",
                "num_stations_impaired",
            ],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset="date", keep="first")
    )

    systemic_station_df = (
        system_membership_df.merge(
            systemic_events_df[CANDIDATE_KEYS],
            on=CANDIDATE_KEYS,
            how="inner",
        )
        .loc[lambda frame: frame["is_impaired"]]
        .copy()
    )

    daily_df = station_day_df.groupby("date", as_index=False).agg(
        num_stations_impaired=("is_impaired", "sum")
    )

    event_details_df = systemic_station_df.groupby(
        CANDIDATE_KEYS,
        as_index=False,
    ).agg(
        station_ids=(
            "station_id",
            lambda values: sorted(values.unique().tolist()),
        ),
        auth_rate_pct=("auth_rate_pct", "mean"),
        start_rate_pct=("start_rate_pct", "mean"),
        demand_pct=("demand_pct", "mean"),
    )
    event_details_df["num_stations_systemic"] = event_details_df[
        "station_ids"
    ].str.len()
    event_details_df["correlates"] = (
        event_details_df["correlate_feature"].astype(str)
        + ": "
        + event_details_df["correlate_value"].astype(str)
    )

    daily_df = daily_df.merge(
        event_details_df[
            [
                "date",
                "num_stations_systemic",
                "event_type",
                "station_ids",
                "correlates",
                "auth_rate_pct",
                "start_rate_pct",
                "demand_pct",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    daily_df["has_systemic_failure"] = daily_df["event_type"].notna()
    daily_df["num_stations_systemic"] = (
        daily_df["num_stations_systemic"].fillna(0).astype(int)
    )

    non_systemic = ~daily_df["has_systemic_failure"]
    nullable_event_columns = [
        "event_type",
        "station_ids",
        "correlates",
        "auth_rate_pct",
        "start_rate_pct",
        "demand_pct",
    ]
    daily_df.loc[
        non_systemic,
        nullable_event_columns,
    ] = None

    rate_columns = [
        "auth_rate_pct",
        "start_rate_pct",
        "demand_pct",
    ]
    daily_df[rate_columns] = daily_df[rate_columns].round(2)
    daily_df["date"] = daily_df["date"].dt.strftime("%Y-%m-%d")

    output_columns = [
        "date",
        "num_stations_impaired",
        "has_systemic_failure",
        "num_stations_systemic",
        "event_type",
        "station_ids",
        "correlates",
        "auth_rate_pct",
        "start_rate_pct",
        "demand_pct",
    ]
    return daily_df[output_columns]


def detect_systemic_events(station_day_df, *, include_firmware_cohorts=True):
    """Return one event row per date from already flagged station-days.

    Reuse the same station-day table to compare cohort settings. Each run
    adjusts its threshold for its own candidate count. At most one group is
    selected per date; output rates average its impaired stations' rates.
    """
    membership = _build_group_membership(
        station_day_df,
        include_firmware_cohorts=include_firmware_cohorts,
    )
    candidates = _score_groups(station_day_df, membership)
    return _select_daily_events(station_day_df, membership, candidates)


def detect_reliability_events(
    sites_csv, stations_csv, attempts_csv, *, include_firmware_cohorts=True
):
    """Load three CSVs and return one reliability-event row per calendar date.

    Convenience entry point for scripts. The notebook calls the individual
    preparation and detection steps so intermediate metrics remain inspectable.
    """
    tables = load_data(sites_csv, stations_csv, attempts_csv)
    station_days = flag_station_impairment(build_station_days(*tables))
    return detect_systemic_events(
        station_days,
        include_firmware_cohorts=include_firmware_cohorts,
    )
