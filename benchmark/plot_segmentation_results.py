#!/usr/bin/env python3
"""
Process segmentation_filtered.csv to compute and save bootstrap IQM distributions,
normalize metrics, and plot violin plots for segmentation results.
"""
import json
import math
import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from get_data import get_data

# Stub geobench for plot_tools import
gb_dir = Path(__file__).parent
sys.modules["geobench"] = types.SimpleNamespace(GEO_BENCH_DIR=gb_dir)
# Add examples folder to import path
sys.path.append(str(Path(__file__).parent))
from plot_tools import bootstrap_iqm
from plot_tools import bootstrap_iqm_aggregate
from plot_tools import make_normalizer
from plot_tools import plot_per_dataset


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data" / "segmentation_filtered.csv"
    output_dir = base_dir / "data" / "segmentation_violin_output"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "violin_plot.pdf"
    training_type = "feature_extraction"

    # Load and rename columns
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = get_data(
            run_name="MarsBenchSegmentationWithSeeds",
            columns=["model_name", "data_name", "seed", "test/iou", "training_type"],
        )
        df = df[df["training_type"] == training_type]
        with open(base_dir / "mappings.json", "r") as f:
            mappings = json.load(f)
        df["model_name"] = df["model_name"].map(mappings["models"])
        df["data_name"] = df["data_name"].map(mappings["data"])

        df = df.rename(columns={"model_name": "model", "data_name": "dataset", "test/iou": "test metric"})
        df["partition name"] = "default"
        df.to_csv(csv_path, index=False)

    datasets = ["aggregated"] + df["dataset"].unique().tolist()

    # Build normalizer and apply
    normalizer = make_normalizer(df, metrics=["test metric"], benchmark_name=None)
    norm_col = normalizer.normalize_data_frame(df, "test metric")

    # Save normalizer mapping
    norm_path = output_dir / "normalizer.json"
    with open(norm_path, "w") as f:
        json.dump(normalizer.range_dict, f, indent=2)

    # Compute bootstrap distributions
    aggregated = bootstrap_iqm_aggregate(df, metric=norm_col, repeat=100)
    per_dataset = bootstrap_iqm(df, metric=norm_col, repeat=100)

    # Save processed data
    aggregated.to_csv(output_dir / "aggregated_bootstrap.csv", index=False)
    per_dataset.to_csv(output_dir / "per_dataset_bootstrap.csv", index=False)

    # Combine for plotting
    combined = pd.concat([aggregated, per_dataset], ignore_index=True)

    # Define model order and colors
    model_order = list(df["model"].unique())
    model_colors = dict(zip(model_order, sns.color_palette("colorblind", len(model_order))))

    num_datasets = len(combined["dataset"].unique())
    max_cols = 5
    num_cols = min(max_cols, num_datasets)
    num_rows = math.ceil(num_datasets / num_cols)
    fig_size = (num_cols * 3, num_rows * 3)  # moderate height per row
    plot_per_dataset(
        combined,
        model_order,
        metric=norm_col,
        model_colors=model_colors,
        datasets=datasets,
        fig_size=fig_size,
        n_legend_rows=1,
    )
    # Rotate x-axis labels and adjust layout
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        # Increase y-axis label title font size
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
        # Format legend: single row and larger font
        legend = ax.get_legend()
        if legend:
            legend.set_ncols(len(model_order))
            legend.set_title("", prop={"size": 14})
            for text in legend.get_texts():
                text.set_fontsize(14)
    # plt.tight_layout()
    # Save figure
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Generated violin plot: {plot_path}")


if __name__ == "__main__":
    main()
