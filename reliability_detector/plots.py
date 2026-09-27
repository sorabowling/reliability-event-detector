"""Reusable figures. Each function returns (figure, axes) without showing or saving."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

EVENT_COLORS = {
    "reader-associated": "#d95f02",
    "site-associated": "#1b9e77",
    "firmware-associated": "#7570b3",
    "site-type-associated": "#e7298a",
}


def _event_dates(daily_df):
    """Copy daily output and convert its date column for plotting."""
    return daily_df.assign(date=pd.to_datetime(daily_df["date"]))


def plot_network_reliability(daily_df):
    """Overlay the selected systemic subset on the total impaired-station count."""
    daily_plot_df = _event_dates(daily_df)
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(
        daily_plot_df["date"],
        daily_plot_df["num_stations_impaired"],
        color="#d1d5db",
        width=0.9,
        label="All impaired stations",
    )

    for event_type, color in EVENT_COLORS.items():
        event_mask = daily_plot_df["event_type"].eq(event_type)
        if not event_mask.any():
            continue
        ax.bar(
            daily_plot_df.loc[event_mask, "date"],
            daily_plot_df.loc[
                event_mask,
                "num_stations_systemic",
            ],
            color=color,
            width=0.9,
            label=event_type,
        )

    ax.set(
        title="Network impairment and detected systemic events",
        xlabel="Date",
        ylabel="Number of stations",
    )
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylim(0, max(5, 1.3 * daily_plot_df["num_stations_impaired"].max()))
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax


def plot_systemic_stations(daily_df, stations_df):
    """Show selected systemic station-days, ordered by reader, site, and station."""
    daily_plot_df = _event_dates(daily_df)
    station_order = stations_df.sort_values(["reader_type", "site_id", "station_id"])[
        "station_id"
    ].tolist()
    date_order = pd.date_range(
        daily_plot_df["date"].min(),
        daily_plot_df["date"].max(),
    )

    systemic_station_dates_df = daily_plot_df.loc[
        daily_plot_df["has_systemic_failure"],
        ["date", "event_type", "station_ids"],
    ].explode("station_ids")

    event_code_by_type = {
        "site-associated": 1,
        "reader-associated": 2,
        "firmware-associated": 3,
        "site-type-associated": 4,
    }
    systemic_station_dates_df["event_code"] = systemic_station_dates_df[
        "event_type"
    ].map(event_code_by_type)

    systemic_matrix_df = systemic_station_dates_df.pivot_table(
        index="station_ids",
        columns="date",
        values="event_code",
        aggfunc="max",
        fill_value=0,
    ).reindex(
        index=station_order,
        columns=date_order,
        fill_value=0,
    )

    heatmap_colors = [
        "#ffffff",
        EVENT_COLORS["site-associated"],
        EVENT_COLORS["reader-associated"],
        EVENT_COLORS["firmware-associated"],
        EVENT_COLORS["site-type-associated"],
    ]

    fig, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(
        systemic_matrix_df,
        cmap=ListedColormap(heatmap_colors),
        vmin=0,
        vmax=4,
        cbar=False,
        xticklabels=7,
        yticklabels=5,
        linewidths=0,
        ax=ax,
    )
    ax.set(
        title="Systemic station-days by event type",
        xlabel="Date",
        ylabel="Station ID (ordered by reader type and site)",
    )
    ax.set_xticklabels(
        [
            pd.to_datetime(label.get_text()).strftime("%b %d")
            for label in ax.get_xticklabels()
        ],
        rotation=45,
        ha="right",
    )
    heatmap_legend = [
        Patch(
            color=EVENT_COLORS[event_type],
            label=event_type,
        )
        for event_type in event_code_by_type
        if event_type in systemic_station_dates_df["event_type"].unique()
    ]
    ax.legend(
        handles=heatmap_legend,
        title="Systemic event",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        frameon=False,
    )
    plt.tight_layout()

    return fig, ax


def plot_firmware_start_rates(
    firmware_start_daily_df,
    firmware_event_start,
    firmware_event_end,
    firmware_correlate,
    *,
    context_days=7,
):
    """Plot daily conditional start rates around one supplied event interval."""
    if pd.isna(firmware_event_start) or pd.isna(firmware_event_end):
        raise ValueError("Choose a detected firmware event interval before plotting.")
    firmware_event_start = pd.Timestamp(firmware_event_start)
    firmware_event_end = pd.Timestamp(firmware_event_end)
    firmware_window_start = firmware_event_start - pd.Timedelta(days=context_days)
    firmware_window_end = firmware_event_end + pd.Timedelta(days=context_days)
    firmware_window_df = firmware_start_daily_df.loc[
        firmware_start_daily_df["date"].between(
            firmware_window_start,
            firmware_window_end,
        )
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.lineplot(
        data=firmware_window_df,
        x="date",
        y="start_rate_pct",
        hue="firmware_version",
        marker="o",
        linewidth=2.5,
        ax=ax,
    )
    ax.axvspan(
        firmware_event_start - pd.Timedelta(hours=12),
        firmware_event_end + pd.Timedelta(hours=12),
        color=EVENT_COLORS["firmware-associated"],
        alpha=0.15,
        label="Detected firmware event",
    )
    ax.text(
        firmware_event_start + (firmware_event_end - firmware_event_start) / 2,
        6,
        f"{firmware_correlate} cohort",
        color=EVENT_COLORS["firmware-associated"],
        ha="center",
        va="bottom",
        fontweight="bold",
    )
    ax.set(
        title=(f"Daily start success by firmware: {firmware_correlate}"),
        xlabel="Date",
        ylabel="Start success rate (%)",
        ylim=(0, 105),
    )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Firmware version",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax


def plot_demand_vs_energy(network_daily_metrics_df):
    """Compare network attempt and energy ratios against their weekday baselines."""
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=network_daily_metrics_df,
        x="demand_pct",
        y="delivered_kwh_pct",
        hue="day_type",
        size="num_stations_impaired",
        sizes=(35, 350),
        palette={
            "no systemic event": "#9ca3af",
            **EVENT_COLORS,
        },
        alpha=0.8,
        ax=ax,
    )
    ax.axvline(100, color="#374151", linestyle="--", linewidth=1)
    ax.axhline(100, color="#374151", linestyle="--", linewidth=1)
    ax.set(
        title="Higher attempt volume does not imply higher energy delivery",
        xlabel="Attempts as percent of expected",
        ylabel="Delivered kWh as percent of expected",
    )
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_label_map = {
        "day_type": "Systemic event",
        "num_stations_impaired": "Impaired stations",
    }
    ax.legend(
        legend_handles,
        [legend_label_map.get(label, label) for label in legend_labels],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    plt.tight_layout()

    return fig, ax


def plot_attempt_outcomes(daily_outcome_pct_df, daily_df):
    """Show daily outcome shares with markers on selected systemic-event dates."""
    daily_plot_df = _event_dates(daily_df)
    outcome_order = daily_outcome_pct_df.columns.tolist()
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.stackplot(
        daily_outcome_pct_df.index,
        *[daily_outcome_pct_df[column] for column in outcome_order],
        labels=outcome_order,
        colors=["#4c78a8", "#e45756", "#f2cf5b"],
        alpha=0.9,
    )
    systemic_dates = daily_plot_df.loc[
        daily_plot_df["has_systemic_failure"],
        "date",
    ]
    ax.scatter(
        systemic_dates,
        np.repeat(99, len(systemic_dates)),
        marker="|",
        s=100,
        color="#111827",
        clip_on=True,
        label="systemic event date",
    )
    ax.set(
        title="Daily attempt-outcome composition",
        xlabel="Date",
        ylabel="Share of attempts (%)",
        ylim=(0, 100),
    )
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(frameon=False, ncol=4, loc="lower left")
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig, ax


def plot_episode_shortfalls(episode_metrics_df, *, top_n=10):
    """Rank the largest event episodes by estimated energy shortfall."""
    top_episodes_df = episode_metrics_df.nlargest(
        top_n,
        "estimated_kwh_shortfall",
    ).sort_values("estimated_kwh_shortfall")

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(
        top_episodes_df["label"],
        top_episodes_df["estimated_kwh_shortfall"] / 1000,
        color=[
            EVENT_COLORS[event_type] for event_type in top_episodes_df["event_type"]
        ],
    )
    ax.set(
        title="Largest estimated energy shortfalls by event episode",
        xlabel="Estimated energy shortfall (MWh)",
        ylabel="",
    )
    legend_handles = [
        Patch(color=color, label=event_type)
        for event_type, color in EVENT_COLORS.items()
        if event_type in top_episodes_df["event_type"].values
    ]
    ax.legend(handles=legend_handles, frameon=False)
    plt.tight_layout()

    return fig, ax


def plot_correlate_shortfalls(correlate_impact_df):
    """Plot shortfall by association and its cumulative share."""
    fig, ax = plt.subplots(figsize=(13, 7))
    x_positions = np.arange(len(correlate_impact_df))
    ax.bar(
        x_positions,
        correlate_impact_df["shortfall_mwh"],
        color=[
            EVENT_COLORS[event_type] for event_type in correlate_impact_df["event_type"]
        ],
    )
    ax.set(
        title="Estimated kWh-shortfall Pareto by systemic correlate",
        xlabel="Systemic correlate",
        ylabel="Estimated shortfall (MWh)",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        correlate_impact_df["correlates"],
        rotation=35,
        ha="right",
    )

    cumulative_ax = ax.twinx()
    cumulative_ax.plot(
        x_positions,
        correlate_impact_df["cumulative_pct"],
        color="#111827",
        marker="o",
        linewidth=2,
    )
    cumulative_ax.axhline(
        80,
        color="#6b7280",
        linestyle="--",
        linewidth=1,
    )
    cumulative_ax.set(
        ylabel="Cumulative share (%)",
        ylim=(0, 105),
    )
    plt.tight_layout()

    return fig, ax
