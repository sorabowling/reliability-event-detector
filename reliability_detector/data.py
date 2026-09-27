"""Load logs and build complete station-day metrics and weekday baselines.

All listed stations are treated as deployed throughout the observed date range.
Baselines use the full window, including the evaluated day and future dates.
"""

import numpy as np
import pandas as pd


def load_data(sites_csv, stations_csv, attempts_csv):
    """Read the three CSVs and return (sites, stations, attempts) DataFrames.

    Attempts include a normalized date and outcome indicator columns.
    """
    sites_df = pd.read_csv(sites_csv).rename(
        columns={
            "type": "site_type",
            "stations": "total_stations_site",
        }
    )
    stations_df = pd.read_csv(stations_csv)
    attempts_df = pd.read_csv(
        attempts_csv,
        parse_dates=[
            "attempt_timestamp",
            "session_start_timestamp",
            "session_end_timestamp",
        ],
    )

    attempts_df["date"] = attempts_df["attempt_timestamp"].dt.normalize()
    attempts_df["is_session"] = attempts_df["attempt_outcome"].eq("session")
    attempts_df["is_auth_error"] = attempts_df["attempt_outcome"].eq("auth error")
    attempts_df["is_start_error"] = attempts_df["attempt_outcome"].eq("start error")

    return sites_df, stations_df, attempts_df


def build_station_days(sites_df, stations_df, attempts_df):
    """Aggregate loaded logs, retaining zero-attempt days and undefined rates.

    Expects the tables returned by load_data. Attempt counts preserve the
    distinct-timestamp convention. Start rates exclude authorization errors.
    Adds power metrics and same-weekday attempt and energy medians without
    changing the input DataFrames.
    """
    observed_station_day_df = attempts_df.groupby(
        ["station_id", "date"],
        as_index=False,
    ).agg(
        attempt_count=("attempt_timestamp", "nunique"),
        session_count=("is_session", "sum"),
        auth_error_count=("is_auth_error", "sum"),
        start_error_count=("is_start_error", "sum"),
        total_kwh_delivered=("kwh", "sum"),
        total_duration_min=("duration_min", "sum"),
    )

    all_dates_df = pd.DataFrame(
        {
            "date": pd.date_range(
                attempts_df["date"].min(),
                attempts_df["date"].max(),
            )
        }
    )

    station_day_df = (
        stations_df.merge(all_dates_df, how="cross")
        .merge(
            observed_station_day_df,
            on=["station_id", "date"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            sites_df[["site_id", "site_type"]],
            on="site_id",
            how="left",
            validate="many_to_one",
        )
    )

    daily_value_columns = [
        "attempt_count",
        "session_count",
        "auth_error_count",
        "start_error_count",
        "total_kwh_delivered",
        "total_duration_min",
    ]
    station_day_df[daily_value_columns] = station_day_df[daily_value_columns].fillna(0)

    station_day_df["day_of_week"] = station_day_df["date"].dt.dayofweek + 1
    station_day_df["expected_attempt_count"] = station_day_df.groupby(
        ["station_id", "day_of_week"]
    )["attempt_count"].transform("median")

    station_day_df["auth_rate_pct"] = np.where(
        station_day_df["attempt_count"].gt(0),
        100
        * (station_day_df["attempt_count"] - station_day_df["auth_error_count"])
        / station_day_df["attempt_count"],
        np.nan,
    )

    observable_start_count = (
        station_day_df["session_count"] + station_day_df["start_error_count"]
    )
    station_day_df["start_rate_pct"] = np.where(
        observable_start_count.gt(0),
        100 * station_day_df["session_count"] / observable_start_count,
        np.nan,
    )
    station_day_df["demand_pct"] = np.where(
        station_day_df["expected_attempt_count"].gt(0),
        100
        * station_day_df["attempt_count"]
        / station_day_df["expected_attempt_count"],
        np.nan,
    )

    session_df = attempts_df.loc[attempts_df["attempt_outcome"].eq("session")].copy()
    session_df["power_kw"] = np.where(
        session_df["duration_min"].gt(0),
        60 * session_df["kwh"] / session_df["duration_min"],
        np.nan,
    )
    station_day_power_df = session_df.groupby(
        ["station_id", "date"],
        as_index=False,
    ).agg(
        median_power_kw=("power_kw", "median"),
        zero_kwh_session_count=(
            "kwh",
            lambda values: values.eq(0).sum(),
        ),
    )
    station_day_df = station_day_df.merge(
        station_day_power_df,
        on=["station_id", "date"],
        how="left",
        validate="one_to_one",
    )
    station_day_df["zero_kwh_session_count"] = station_day_df[
        "zero_kwh_session_count"
    ].fillna(0)

    station_day_df["expected_kwh_delivered"] = station_day_df.groupby(
        ["station_id", "day_of_week"]
    )["total_kwh_delivered"].transform("median")
    return station_day_df
