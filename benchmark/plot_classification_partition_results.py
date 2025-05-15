"""
Process classification_partition_filtered.csv to compute and save bootstrap IQM distributions,
normalize metrics, and plot both violin and line plots across partitions.
"""
import json
import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from get_data import get_data

sns.set_style("dark", {"grid.color": "0.98", "axes.facecolor": "(0.95, 0.95, 0.97)"})

# Stub geobench for plot_tools import
gb_dir = Path(__file__).parent
sys.modules["geobench"] = types.SimpleNamespace(GEO_BENCH_DIR=gb_dir)
# Add examples folder to import path
sys.path.append(str(Path(__file__).parent))
from plot_tools import make_normalizer


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data" / "classification_partition_filtered.csv"
    output_dir = base_dir / "data" / "classification_partition_violin_output"
    full_data_path = base_dir / "data" / "classification_filtered.csv"
    plot_path = output_dir / "lineplot_partitions_row.pdf"
    output_dir.mkdir(exist_ok=True)

    # Load and rename columns
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = get_data(
            run_name="MarsBenchClassificationWithPartitionSeeds",
            columns=["model_name", "data_name", "seed", "test/F1Score_weighted", "training_type", "partition"],
        )
        df = df[df["training_type"] == "feature_extraction"]
        with open(base_dir / "mappings.json", "r") as f:
            mappings = json.load(f)
        df["model_name"] = df["model_name"].map(mappings["models"])
        df["data_name"] = df["data_name"].map(mappings["data"])

        df = df.rename(
            columns={
                "model_name": "model",
                "data_name": "dataset",
                "test/F1Score_weighted": "test metric",
                "partition": "partition name",
            }
        )
        full_data = pd.read_csv(full_data_path)
        full_data["partition name"] = 1.0
        df = pd.concat([df, full_data[df.columns]], ignore_index=True).sort_values(
            by=["model", "dataset", "partition name"]
        )
        df.to_csv(csv_path, index=False)

    datasets = ["aggregated"] + [
        "mb-atmospheric_dust_cls_edr",
        "mb-domars16k",
        "mb-change_cls_hirise",
        "mb-frost_cls",
        "mb-surface_cls",
        "mb-surface_multi_label_cls",
    ]

    normalizer = make_normalizer(df, metrics=["test metric"], benchmark_name=None)
    norm_col = normalizer.normalize_data_frame(df, "test metric")

    # Save normalizer mapping
    norm_path = output_dir / "normalizer.json"
    with open(norm_path, "w") as f:
        json.dump(normalizer.range_dict, f, indent=2)

    # Skip bootstrapping and use normalized data directly
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
    combined_stats.to_csv(output_dir / "partition_stats.csv", index=False)

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

        # Remove legend from all but the last subplot
        if i < len(datasets) - 1 and hasattr(ax, "get_legend") and ax.get_legend():
            ax.get_legend().remove()

    # Format axes to match plot_segmentation_results.py style
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
    from matplotlib.patches import Patch

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
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plots in {plot_path}")


if __name__ == "__main__":
    main()
