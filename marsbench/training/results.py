"""
Utilities for saving model results and predictions.
"""

import datetime
import json
import logging
import os
from typing import Dict
from typing import List

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def save_benchmark_results(cfg: DictConfig, results: List):
    """Save benchmark results to JSON file.

    Args:
        cfg: Configuration object
        results: Benchmark results to save
    """
    if not cfg.get("output_path"):
        log.warning("No output_path specified in config, skipping benchmark results saving")
        return

    # Create benchmark directory
    benchmark_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.output_path, "benchmarks")
    os.makedirs(benchmark_dir, exist_ok=True)

    # Prefix (model name + dataset)
    prefix = f"benchmark_{cfg.model.name}_{cfg.data_name}"

    # Consolidated results file
    results_file = os.path.join(benchmark_dir, f"{prefix}_results.csv")

    # Current timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare entry for this run
    entry = {"timestamp": timestamp, "model": cfg.model.name, "dataset": cfg.data_name, "results": results[0]}

    # Load existing results or create new file
    try:
        if os.path.exists(results_file):
            all_results = pd.read_csv(results_file)
        else:
            all_results = pd.DataFrame()

        # Add new entry
        all_results = pd.concat([all_results, pd.DataFrame([entry])], ignore_index=True)

        # Write back the updated file
        all_results.to_csv(results_file, index=False)

        log.info(f"Benchmark results added to {results_file}")

    except Exception as e:
        log.error(f"Failed to save to consolidated file: {str(e)}")
        log.info("Falling back to individual file saving")

        # Save to individual file as fallback
        individual_file = os.path.join(benchmark_dir, f"{prefix}_{timestamp}.csv")
        pd.DataFrame([entry]).to_csv(individual_file, index=False)

        log.info(f"Benchmark results saved to individual file: {individual_file}")


def save_predictions(cfg: DictConfig, predictions: List[Dict]):
    """Save model predictions to files.

    Args:
        cfg: Configuration object
        predictions: Prediction results to save
    """
    if not cfg.get("output_path") and not cfg.get("prediction_output_path"):
        log.warning("No output paths specified in config, skipping predictions saving")
        return

    try:
        # Create predictions directory with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine base output directory
        if cfg.get("prediction_output_path"):
            # Use custom prediction output path if provided
            base_dir = cfg.prediction_output_path
            if not os.path.isabs(base_dir):
                # Convert to absolute path if relative
                base_dir = os.path.join(hydra.utils.get_original_cwd(), base_dir)
            log.info(f"Using custom prediction output path: {base_dir}")
        else:
            # Default to output_path/predictions
            base_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.output_path, "predictions")

        # Create final directory for this prediction run
        pred_dir = os.path.join(base_dir, f"{cfg.model.name}_{timestamp}")
        os.makedirs(pred_dir, exist_ok=True)
        log.info(f"Saving predictions to: {pred_dir}")

        # Concatenate all batch predictions
        all_probs = torch.cat([batch["probabilities"] for batch in predictions])
        all_preds = torch.cat([batch["predictions"] for batch in predictions])

        # Save as CSV for easy analysis
        df = pd.DataFrame(all_probs.cpu().numpy())
        df.columns = [f"class_{i}" for i in range(df.shape[1])]
        df["predicted_class"] = all_preds.cpu().numpy()

        # Save outputs in different formats
        df.to_csv(os.path.join(pred_dir, "predictions.csv"), index=False)

        # Save as NumPy arrays
        np.save(os.path.join(pred_dir, "probabilities.npy"), all_probs.cpu().numpy())
        np.save(os.path.join(pred_dir, "predictions.npy"), all_preds.cpu().numpy())

        # Save configuration for reproducibility
        with open(os.path.join(pred_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

        # Save metadata
        with open(os.path.join(pred_dir, "metadata.json"), "w") as f:
            metadata = {
                "model_name": cfg.model.name,
                "data_name": cfg.data_name,
                "timestamp": timestamp,
                "config": hydra.utils.to_yaml(cfg),
            }
            json.dump(metadata, f, indent=2)

        log.info(f"Predictions saved successfully to {pred_dir}")
    except Exception as e:
        log.error(f"Failed to save predictions: {str(e)}")
