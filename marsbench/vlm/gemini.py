#!/usr/bin/env python3
import logging
import os
from itertools import islice

import hydra
import pandas as pd
import torch
from google import genai
from google.genai import types
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torchvision.transforms import ToPILImage

from marsbench.data.mars_datamodule import MarsDataModule
from marsbench.utils.config_mapper import load_dynamic_configs
from marsbench.utils.load_mapping import get_class_idx
from marsbench.utils.load_mapping import get_class_name

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def run_gemini(cfg: DictConfig):
    # load dynamic configs (data, model, prompts)
    cfg = load_dynamic_configs(cfg)
    logger.info(f"Loaded dynamic configs. VLM: {cfg.vlm}")

    # prepare data
    data_module = MarsDataModule(cfg)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # determine batch limit
    max_batches = getattr(cfg.vlm, "max_batches", 2)
    total_batches = len(test_loader)
    limit = max_batches if max_batches and max_batches < total_batches else total_batches
    logger.info(f"Processing {limit}/{total_batches} batches.")

    # setup Gemini client
    api_key = (
        cfg.vlm.api_key if hasattr(cfg.vlm, "api_key") and cfg.vlm.api_key is not None else os.getenv("GOOGLE_API_KEY")
    )
    client = genai.Client(api_key=api_key)
    model_name = cfg.vlm.model_name
    safety = [types.SafetySetting(category=s["category"], threshold=s["threshold"]) for s in cfg.vlm.safety_settings]

    to_pil = ToPILImage()
    results = []
    # limit to first max_batches batches
    for batch_idx, (imgs, labels) in islice(enumerate(test_loader), limit):
        for idx in range(imgs.size(0)):
            img_tensor = imgs[idx]
            true_label = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]
            pil_img = to_pil(img_tensor.cpu())

            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[cfg.prompts.user_prompt, pil_img],
                    config=types.GenerateContentConfig(
                        system_instruction=cfg.prompts.system_instructions,
                        temperature=cfg.vlm.temperature,
                        safety_settings=safety,
                    ),
                )
                pred = resp.text.strip()
                err = ""
            except Exception as e:
                pred = ""
                err = str(e)

            results.append(
                {
                    "batch": batch_idx,
                    "index": idx,
                    "true_label_idx": true_label,
                    "pred_label_idx": get_class_idx(pred, cfg),
                    "true_label": get_class_name(true_label, cfg),
                    "pred_label": pred,
                    "error": err,
                }
            )

    # save predictions
    out_dir = os.path.join(get_original_cwd(), cfg.output_path, cfg.task, cfg.data_name, cfg.vlm.name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "predictions.csv")
    logger.info(f"Saving {len(results)} predictions to {out_csv}")
    # save predictions using pandas
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved VLM predictions to: {out_csv}")


if __name__ == "__main__":
    run_gemini()
