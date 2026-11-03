
"""
Configuration loader for the 2.5D LiDAR Mapping project.
"""

from pathlib import Path
from typing import Any

import yaml


# Resolve the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "mapping.yaml"


def load_config(
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load and validate the project's YAML configuration.

    Args:
        config_path:
            Optional path to a custom YAML configuration.

    Returns:
        Dictionary containing configuration sections.
    """

    path = (
        Path(config_path)
        if config_path is not None
        else DEFAULT_CONFIG_PATH
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration must contain a YAML mapping."
        )

    required_sections = [
        "project",
        "mapping",
        "semantic",
        "temporal",
        "tracking",
        "evaluation",
        "data",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing configuration section: {section}"
            )

        if not isinstance(config[section], dict):
            raise ValueError(
                f"Configuration section must be a mapping: {section}"
            )

    mapping = config["mapping"]

    if not (
        mapping["x_min"] < mapping["x_max"]
        and mapping["y_min"] < mapping["y_max"]
    ):
        raise ValueError("Invalid mapping boundaries.")

    resolutions = [
        mapping["near_resolution"],
        mapping["mid_resolution"],
        mapping["far_resolution"],
    ]

    if not all(resolution > 0 for resolution in resolutions):
        raise ValueError("Grid resolutions must be positive.")

    if not (
        resolutions[0] <= resolutions[1] <= resolutions[2]
    ):
        raise ValueError(
            "Resolutions must progress from fine to coarse."
        )

    if not (
        0 < mapping["near_distance"] < mapping["mid_distance"]
    ):
        raise ValueError("Invalid distance thresholds.")

    threshold = config["semantic"]["confidence_threshold"]

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Semantic confidence threshold must be between 0 and 1."
        )

    if config["temporal"]["occupancy_persistence_frames"] < 1:
        raise ValueError(
            "Occupancy persistence must be at least 1 frame."
        )

    return config


def get_config_value(
    section: str,
    key: str,
    config: dict[str, Any] | None = None,
) -> Any:
    """
    Retrieve a specific configuration value.
    """

    if config is None:
        config = load_config()

    try:
        return config[section][key]
    except KeyError as exc:
        raise KeyError(
            f"Configuration value not found: {section}.{key}"
        ) from exc
