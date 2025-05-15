#!/usr/bin/env python3
"""
Process earth_data.csv to normalize metrics and plot line plots across partitions.
"""

import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Set plot style to match other plots
sns.set_style("dark", {"grid.color": "0.98", "axes.facecolor": "(0.95, 0.95, 0.97)"})

# Stub geobench for plot_tools import
gb_dir = Path(__file__).parent
sys.modules["geobench"] = types.SimpleNamespace(GEO_BENCH_DIR=gb_dir)
# Add examples folder to import path
sys.path.append(str(Path(__file__).parent))
from plot_tools import make_normalizer


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data" / "earth_data.csv"
    output_dir = base_dir / "data" / "earth_data_output"
    plot_path = output_dir / "earth_data_line_plot.pdf"
    output_dir.mkdir(exist_ok=True)

    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Get unique datasets
    datasets = ["aggregated"] + sorted(df["dataset"].unique().tolist())

    # Normalize metrics
    normalizer = make_normalizer(df, metrics=["test metric"])
    norm_col = normalizer.normalize_data_frame(df, "test metric")

    # Calculate mean and std per dataset, model, partition group
    stats_df = df.groupby(["dataset", "model", "partition name"])[norm_col].agg(["mean", "std", "count"]).reset_index()

    # Calculate aggregated statistics across all datasets
    # First, get the mean of normalized metrics per model/partition/seed combination
    model_partition_means = df.groupby(["model", "partition name", "seed"])[norm_col].mean().reset_index()

    # Then calculate statistics on these means
    aggregated_stats = (
        model_partition_means.groupby(["model", "partition name"])["normalized test metric"]
        .agg([("mean", "mean"), ("std", "std"), ("count", "count")])
        .reset_index()
    )

    # Add dataset column with 'aggregated' value
    aggregated_stats["dataset"] = "aggregated"

    # Reorder columns to match stats_df
    aggregated_stats = aggregated_stats[["dataset", "model", "partition name", "mean", "std", "count"]]

    # Combine per-dataset and aggregated stats
    combined_stats = pd.concat([stats_df, aggregated_stats], ignore_index=True)

    # Save processed stats for reference
    combined_stats.to_csv(output_dir / "earth_data_stats.csv", index=False)

    # Convert partition_name to float for proper plotting
    combined_stats["partition_value"] = combined_stats["partition name"].astype(float)

    # Create figure with one row of subplots
    fig, axes = plt.subplots(1, len(datasets), figsize=(len(datasets) * 3, 4), sharey=True)

    # Process each dataset in its own subplot
    for i, dataset in enumerate(datasets):
        ax = axes[i] if len(datasets) > 1 else axes  # Handle single subplot case
        if dataset == "aggregated":
            ax.set_facecolor("#cff6fc")

        # Get data for this dataset
        ds_data = combined_stats[combined_stats["dataset"] == dataset]

        # Plot each model's data
        for model in ds_data["model"].unique():
            # Get data for this model
            model_data = ds_data[ds_data["model"] == model].sort_values("partition_value")

            # Plot the line
            line = ax.plot(model_data["partition_value"], model_data["mean"], markersize=8, linewidth=1.5, label=model)[
                0
            ]
            color = line.get_color()

            # Add confidence band (mean ± std)
            ax.fill_between(
                model_data["partition_value"],
                model_data["mean"] - model_data["std"],
                model_data["mean"] + model_data["std"],
                alpha=0.3,
                color=color,
            )

        # Set log scale on x-axis and format as percentages
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{float(x):.0%}"))

        # Format subplot
        ax.grid(True, linestyle="--", axis="y", alpha=0.7)
        ax.set_xlabel(dataset, fontsize=14)

        # Only set ylabel on first subplot
        if i == 0:
            ax.set_ylabel("normalized test metric", fontsize=14)

    # Format axes to match other plots
    for ax in fig.axes:
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)

    # Remove existing legends from all subplots
    for ax in fig.axes:
        if hasattr(ax, "get_legend") and ax.get_legend():
            ax.get_legend().remove()

    # Get unique models and their colors
    unique_models = combined_stats["model"].unique()
    model_colors = {}

    # Extract colors from the plotted lines
    for ax in fig.axes:
        for line in ax.get_lines():
            if hasattr(line, "get_label") and line.get_label() in unique_models:
                model_colors[line.get_label()] = line.get_color()

    # Create custom legend handles with rectangle patches
    custom_handles = [
        Patch(facecolor=model_colors.get(model, "gray"), edgecolor="black", label=model) for model in unique_models
    ]

    # Add a single unified legend at the top of the figure with rectangle patches
    fig.legend(
        custom_handles,
        unique_models,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(unique_models),
        fontsize=14,
    )

    # Adjust layout and save
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved Earth data plot to {plot_path}")


if __name__ == "__main__":
    main()
