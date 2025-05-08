#!/usr/bin/env python3
import base64
import io
import logging
import os
from itertools import islice

import hydra
import openai
import pandas as pd
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torchvision.transforms import ToPILImage

from marsbench.data.mars_datamodule import MarsDataModule
from marsbench.utils.config_mapper import load_dynamic_configs
from marsbench.utils.load_mapping import get_class_idx
from marsbench.utils.load_mapping import get_class_name

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.1", config_path="../configs", config_name="config")
def run_chatgpt(cfg: DictConfig):
    # configure logging
    logging.basicConfig(level=logging.INFO)

    # load configs
    cfg = load_dynamic_configs(cfg)
    logger.info(f"Loaded dynamic configs. VLM: {cfg.vlm}")

    # setup data
    data_module = MarsDataModule(cfg)
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # determine batch limit
    max_batches = getattr(cfg.vlm, "max_batches", 2)
    total_batches = len(test_loader)
    limit = max_batches if max_batches and max_batches < total_batches else total_batches
    logger.info(f"Processing {limit}/{total_batches} batches.")

    # setup OpenAI client
    api_key = cfg.vlm.api_key if hasattr(cfg.vlm, "api_key") and cfg.vlm.api_key else os.getenv("OPENAI_API_KEY")
    openai.api_key = api_key
    model_name = cfg.vlm.model_name

    to_pil = ToPILImage()
    results = []

    # iterate
    batch_iter = islice(enumerate(test_loader), max_batches) if max_batches else enumerate(test_loader)
    for batch_idx, (imgs, labels) in batch_iter:
        for idx in range(imgs.size(0)):
            img_tensor = imgs[idx]
            true_label = labels[idx].item() if isinstance(labels[idx], torch.Tensor) else labels[idx]
            pil_img = to_pil(img_tensor.cpu())

            # downsize & compress to reduce token size
            resize_cfg = getattr(cfg.vlm, "image_size", None)
            if resize_cfg:
                pil_img = pil_img.resize(tuple(resize_cfg))
            buf = io.BytesIO()
            quality = getattr(cfg.vlm, "jpeg_quality", None)
            if quality is not None:
                pil_img.save(buf, format="JPEG", quality=quality)
            else:
                pil_img.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            # build messages
            user_content = f"{cfg.prompts.user_prompt}\n![](data:image/png;base64,{img_b64})"
            messages = [
                {"role": "system", "content": cfg.prompts.system_instructions},
                {"role": "user", "content": user_content},
            ]

            try:
                resp = openai.ChatCompletion.create(
                    model=model_name,
                    messages=messages,
                    temperature=cfg.vlm.temperature,
                )
                pred = resp.choices[0].message.content.strip()
                err = ""
            except Exception as e:
                pred = ""
                err = str(e)
                logger.error(f"Prediction error at batch {batch_idx}, index {idx}: {err}")

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

    # save
    out_dir = os.path.join(get_original_cwd(), cfg.output_path, cfg.task, cfg.data_name, cfg.vlm.name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "predictions_chatgpt.csv")
    logger.info(f"Saving {len(results)} predictions to {out_csv}")
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    logger.info(f"Saved ChatGPT predictions to: {out_csv}")


if __name__ == "__main__":
    run_chatgpt()
