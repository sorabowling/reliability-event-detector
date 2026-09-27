"""Build firmware comparisons, network metrics, and reference-based energy estimates."""

import numpy as np
import pandas as pd


def summarize_events(daily_df):
    """Count selected event days and summarize their station-level rates."""
    event_summary_df = (
        daily_df.loc[daily_df["has_systemic_failure"]]
        .groupby(["event_type", "correlates"], as_index=False)
        .agg(
            systemic_days=("date", "count"),
            peak_stations_systemic=(
                "num_stations_systemic",
                "max",
            ),
            mean_auth_rate_pct=("auth_rate_pct", "mean"),
            mean_start_rate_pct=("start_rate_pct", "mean"),
            mean_demand_pct=("demand_pct", "mean"),
        )
        .sort_values("systemic_days", ascending=False)
    )

    return event_summary_df


def network_metrics(station_metrics_df, daily_df):
    """Compare summed attempts and energy against summed station weekday baselines."""
    network_daily_metrics_df = station_metrics_df.groupby("date", as_index=False).agg(
        attempts=("attempt_count", "sum"),
        expected_attempts=(
            "expected_attempt_count",
            "sum",
        ),
        delivered_kwh=("total_kwh_delivered", "sum"),
        expected_kwh=("expected_kwh_delivered", "sum"),
    )
    network_daily_metrics_df["demand_pct"] = (
        100
        * network_daily_metrics_df["attempts"]
        / network_daily_metrics_df["expected_attempts"]
    )
    network_daily_metrics_df["delivered_kwh_pct"] = (
        100
        * network_daily_metrics_df["delivered_kwh"]
        / network_daily_metrics_df["expected_kwh"]
    )

    daily_plot_df = daily_df.copy()
    daily_plot_df["date"] = pd.to_datetime(daily_plot_df["date"])
    daily_plot_df["day_type"] = daily_plot_df["event_type"].fillna("no systemic event")

    network_daily_metrics_df = network_daily_metrics_df.merge(
        daily_plot_df[
            [
                "date",
                "has_systemic_failure",
                "num_stations_impaired",
                "day_type",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
    return network_daily_metrics_df


def firmware_start_rates(attempts_df, stations_df):
    """Pool sessions and observable starts within each firmware version and date."""
    attempts_with_firmware_df = attempts_df.merge(
        stations_df[["station_id", "firmware_version"]],
        on="station_id",
        how="left",
        validate="many_to_one",
    )
    attempts_with_firmware_df["is_session"] = attempts_with_firmware_df[
        "attempt_outcome"
    ].eq("session")
    attempts_with_firmware_df["is_start_error"] = attempts_with_firmware_df[
        "attempt_outcome"
    ].eq("start error")

    firmware_start_daily_df = attempts_with_firmware_df.groupby(
        ["date", "firmware_version"],
        as_index=False,
    ).agg(
        sessions=("is_session", "sum"),
        start_errors=("is_start_error", "sum"),
    )
    firmware_start_daily_df["observable_starts"] = (
        firmware_start_daily_df["sessions"] + firmware_start_daily_df["start_errors"]
    )
    firmware_start_daily_df["start_rate_pct"] = (
        100
        * firmware_start_daily_df["sessions"]
        / firmware_start_daily_df["observable_starts"]
    )

    return firmware_start_daily_df


def firmware_period_rates(
    firmware_start_daily_df, firmware_event_start, firmware_event_end
):
    """Pool counts for the whole window and inside/outside an inclusive event interval."""
    firmware_period_rates = []
    for period_name, period_mask in {
        "Whole period": pd.Series(True, index=firmware_start_daily_df.index),
        "Outside event": ~firmware_start_daily_df["date"].between(
            firmware_event_start, firmware_event_end
        ),
        "During event": firmware_start_daily_df["date"].between(
            firmware_event_start, firmware_event_end
        ),
    }.items():
        period_counts = (
            firmware_start_daily_df.loc[period_mask]
            .groupby("firmware_version")
            .agg(
                sessions=("sessions", "sum"),
                start_errors=("start_errors", "sum"),
            )
        )
        period_rates = (
            100
            * period_counts["sessions"]
            / (period_counts["sessions"] + period_counts["start_errors"])
        )
        firmware_period_rates.append(period_rates.rename(period_name))

    firmware_period_summary_df = pd.concat(firmware_period_rates, axis=1)
    return firmware_period_summary_df


def demand_energy_summary(network_daily_metrics_df):
    """Summarize network volume and energy ratios by selected systemic-event status."""
    demand_energy_summary_df = (
        network_daily_metrics_df.assign(
            systemic_status=np.where(
                network_daily_metrics_df["has_systemic_failure"],
                "Systemic",
                "Non-systemic",
            )
        )
        .groupby("systemic_status", as_index=False)
        .agg(
            mean_demand_pct=("demand_pct", "mean"),
            median_demand_pct=("demand_pct", "median"),
            mean_delivered_kwh_pct=(
                "delivered_kwh_pct",
                "mean",
            ),
            median_delivered_kwh_pct=(
                "delivered_kwh_pct",
                "median",
            ),
        )
    )

    return demand_energy_summary_df


def daily_outcome_percentages(attempts_df):
    """Return daily attempt-outcome shares, retaining dates without recorded attempts."""
    date_order = pd.date_range(attempts_df["date"].min(), attempts_df["date"].max())
    daily_outcomes_df = (
        attempts_df.groupby(["date", "attempt_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(date_order, fill_value=0)
    )
    outcome_order = [
        outcome
        for outcome in ["session", "auth error", "start error"]
        if outcome in daily_outcomes_df.columns
    ]
    daily_outcome_pct_df = 100 * daily_outcomes_df[outcome_order].div(
        daily_outcomes_df[outcome_order].sum(axis=1), axis=0
    )

    return daily_outcome_pct_df


def estimate_energy_shortfall(station_metrics_df, daily_df):
    """Estimate daily shortfall for selected impaired stations and assign episode IDs.

    Clip expected minus actual kWh at zero AFTER summing stations. Consecutive
    dates sharing event type and correlate form an episode. This is a reference
    gap, not causal lost energy or measured unmet customer demand.
    """
    daily_plot_df = daily_df.assign(date=pd.to_datetime(daily_df["date"]))
    systemic_station_days_df = (
        daily_plot_df.loc[
            daily_plot_df["has_systemic_failure"],
            [
                "date",
                "event_type",
                "correlates",
                "num_stations_systemic",
                "station_ids",
            ],
        ]
        .explode("station_ids")
        .rename(columns={"station_ids": "station_id"})
        .merge(
            station_metrics_df[
                [
                    "date",
                    "station_id",
                    "total_kwh_delivered",
                    "expected_kwh_delivered",
                ]
            ],
            on=["date", "station_id"],
            how="left",
            validate="one_to_one",
        )
    )

    event_day_impact_df = (
        systemic_station_days_df.groupby(
            ["date", "event_type", "correlates"],
            as_index=False,
        )
        .agg(
            actual_kwh=("total_kwh_delivered", "sum"),
            expected_kwh=("expected_kwh_delivered", "sum"),
            num_stations_systemic=(
                "num_stations_systemic",
                "first",
            ),
        )
        .sort_values("date")
    )
    event_day_impact_df["estimated_kwh_shortfall"] = (
        event_day_impact_df["expected_kwh"] - event_day_impact_df["actual_kwh"]
    ).clip(lower=0)

    same_episode = (
        event_day_impact_df["event_type"].eq(event_day_impact_df["event_type"].shift())
        & event_day_impact_df["correlates"].eq(
            event_day_impact_df["correlates"].shift()
        )
        & event_day_impact_df["date"].diff().dt.days.eq(1)
    )
    event_day_impact_df["episode_id"] = (~same_episode).cumsum()

    return event_day_impact_df


def summarize_episodes(event_day_impact_df):
    """Aggregate daily energy shortfalls into consecutive selected-group episodes."""
    episode_metrics_df = event_day_impact_df.groupby(
        ["episode_id", "event_type", "correlates"],
        as_index=False,
    ).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        duration_days=("date", "size"),
        peak_stations_systemic=(
            "num_stations_systemic",
            "max",
        ),
        estimated_kwh_shortfall=(
            "estimated_kwh_shortfall",
            "sum",
        ),
    )
    episode_metrics_df["episode"] = np.where(
        episode_metrics_df["start_date"].eq(episode_metrics_df["end_date"]),
        episode_metrics_df["start_date"].dt.strftime("%b %d"),
        episode_metrics_df["start_date"].dt.strftime("%b %d")
        + "–"
        + episode_metrics_df["end_date"].dt.strftime("%b %d"),
    )
    episode_metrics_df["label"] = (
        episode_metrics_df["episode"] + " | " + episode_metrics_df["correlates"]
    )

    return episode_metrics_df


def summarize_correlates(event_day_impact_df):
    """Rank selected associations by energy shortfall and calculate cumulative shares."""
    correlate_impact_df = (
        event_day_impact_df.groupby(
            ["event_type", "correlates"],
            as_index=False,
        )["estimated_kwh_shortfall"]
        .sum()
        .sort_values(
            "estimated_kwh_shortfall",
            ascending=False,
        )
    )
    correlate_impact_df["shortfall_mwh"] = (
        correlate_impact_df["estimated_kwh_shortfall"] / 1000
    )
    correlate_impact_df["cumulative_pct"] = (
        100
        * correlate_impact_df["estimated_kwh_shortfall"].cumsum()
        / correlate_impact_df["estimated_kwh_shortfall"].sum()
    )

    return correlate_impact_df
