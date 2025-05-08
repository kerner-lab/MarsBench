import json
import logging
import os
import re
from typing import Dict

from omegaconf import DictConfig
from omegaconf import ListConfig

logger = logging.getLogger(__name__)


def load_mapping(data_dir: str, num_classes: int | None = None) -> Dict[int, str] | None:
    """
    Load a mapping from a JSON file.
    If num_classes is provided, check if the number of classes in the mapping matches the provided number.

    Args:
        data_dir (str): Path to the data directory.
        (optional) num_classes (int): Number of classes.

    Returns:
        Dict[int, str] | None: Mapping from class indices to class names, or None.
    """
    if os.path.exists(os.path.join(data_dir, "mapping.json")):
        path = os.path.join(data_dir, "mapping.json")
        with open(path, "r") as f:
            raw = json.load(f)
            if isinstance(raw, dict):
                mapping = {int(k): v for k, v in raw.items()}
            elif isinstance(raw, list):
                mapping = {i: v for i, v in enumerate(raw)}
            else:
                logger.error("Mapping JSON root must be dict or list")
                return None
        if num_classes is not None and len(mapping) != num_classes:
            logger.warning(
                f"Number of classes in mapping ({len(mapping)}) does not " f"match num_classes ({num_classes})"
            )
            return None
        logger.info(f"Loaded mapping from {os.path.join(data_dir, 'mapping.json')}")
        return mapping
    else:
        logger.warning("Mapping file not found")
        return None


def get_class_name(class_idx: int, cfg: DictConfig) -> str:
    mapping = getattr(cfg, "mapping", None)
    if mapping:
        name = mapping.get(class_idx, class_idx)
        if isinstance(name, (tuple, list, ListConfig)):
            if not name:
                raise ValueError(f"Empty mapping tuple for class index {class_idx}")
            name = name[0]
        if isinstance(name, (str, int)):
            return str(name)
        raise ValueError(f"Invalid mapping for class index {class_idx}: {name}")
    return str(class_idx)


def get_class_idx(class_name: str, cfg: DictConfig) -> int:
    mapping = getattr(cfg, "mapping", None)
    class_name = re.sub(r"[^A-Za-z]", "", class_name).lower()
    if mapping:
        for idx, name in mapping.items():
            if isinstance(name, (tuple, list, ListConfig)):
                if class_name in name:
                    return idx
            elif isinstance(name, (str, int)):
                if str(name).lower() == class_name:
                    return idx
    logger.warning(f"Class name {class_name} not found in mapping")
    return class_name
