import json
import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_mapping(data_dir: Path, num_classes: int | None = None) -> Dict[int, str] | None:
    """
    Load a mapping from a JSON file.
    If num_classes is provided, check if the number of classes in the mapping matches the provided number.

    Args:
        data_dir (Path): Path to the data directory.
        (optional) num_classes (int): Number of classes.

    Returns:
        Dict[int, str] | None: Mapping from class indices to class names, or None.
    """
    if os.path.exists(data_dir / "mapping.json"):
        with open(data_dir / "mapping.json", "r") as f:
            mapping = {int(k): v.strip().lower().replace(" ", "_").replace("-", "_") for k, v in json.load(f).items()}
        if num_classes is not None and len(mapping) != num_classes:
            logger.warning(
                f"Number of classes in mapping ({len(mapping)}) does not " f"match num_classes ({num_classes})"
            )
            return None
        logger.info(f"Loaded mapping from {data_dir / 'mapping.json'}")
        return mapping
    else:
        logger.warning("Mapping file not found")
        return None
