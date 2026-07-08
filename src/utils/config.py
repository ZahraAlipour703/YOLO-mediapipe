"""
Configuration Manager
=====================

This module loads project configurations from a YAML file and provides
easy access to configuration values throughout the project.

Author:
    Zahra Alipour

Project:
    YOLOv8 + MediaPipe Hand Tracking

Example:
--------
from src.utils.config import Config

cfg = Config("config.yaml")

print(cfg["model"]["weights"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """
    Configuration loader.

    Parameters
    ----------
    config_path : str | Path
        Path to YAML configuration file.
    """

    def __init__(self, config_path: str | Path) -> None:

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found:\n{self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self._config: Dict[str, Any] = yaml.safe_load(file)

    def __getitem__(self, key: str) -> Any:
        """
        Enables dictionary-style access.

        Example
        -------
        cfg["model"]
        """
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Safe getter.

        Parameters
        ----------
        key : str
            Configuration key.

        default : Any
            Default value if key does not exist.
        """
        return self._config.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """
        Return complete configuration.
        """
        return self._config

    def __repr__(self) -> str:
        return f"Config('{self.config_path}')"