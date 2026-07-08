"""
preprocessing.py
================

Dataset preparation utilities for the YOLO + MediaPipe Hand Tracking project.

Responsibilities
----------------
1. Download dataset (optional)
2. Extract ZIP archives
3. Validate dataset structure
4. Create YOLO dataset YAML
5. Count images and labels
6. Print dataset statistics

Author:
Zahra Alipour

"""

from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

LOGGER = logging.getLogger(__name__)


class DatasetPreprocessor:
    """
    Dataset preparation class.

    Parameters
    ----------
    dataset_dir : Path
        Root dataset directory.
    """

    def __init__(self, dataset_dir: str):

        self.dataset_dir = Path(dataset_dir)

        self.train_images = self.dataset_dir / "train/images"
        self.train_labels = self.dataset_dir / "train/labels"

        self.valid_images = self.dataset_dir / "valid/images"
        self.valid_labels = self.dataset_dir / "valid/labels"

        self.test_images = self.dataset_dir / "test/images"
        self.test_labels = self.dataset_dir / "test/labels"

    ####################################################################
    # ZIP EXTRACTION
    ####################################################################

    def extract_zip(
        self,
        zip_path: str,
        output_dir: str
    ) -> None:
        """
        Extract dataset zip.

        Parameters
        ----------
        zip_path : str

        output_dir : str
        """

        zip_path = Path(zip_path)

        output_dir = Path(output_dir)

        LOGGER.info("Extracting dataset...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        LOGGER.info("Extraction completed.")

    ####################################################################
    # VALIDATION
    ####################################################################

    def validate_structure(self) -> bool:
        """
        Check dataset folder structure.
        """

        required = [

            self.train_images,
            self.train_labels,

            self.valid_images,
            self.valid_labels,

            self.test_images,
            self.test_labels

        ]

        missing = []

        for folder in required:

            if not folder.exists():

                missing.append(folder)

        if missing:

            LOGGER.error("Dataset structure is invalid.")

            for folder in missing:
                LOGGER.error(folder)

            return False

        LOGGER.info("Dataset structure validated.")

        return True

    ####################################################################
    # COUNT FILES
    ####################################################################

    @staticmethod
    def count_files(folder: Path, extension: str) -> int:

        return len(list(folder.glob(f"*.{extension}")))

    ####################################################################
    # DATASET STATS
    ####################################################################

    def dataset_statistics(self) -> Dict[str, int]:

        stats = {

            "train_images":
                self.count_files(self.train_images, "jpg"),

            "train_labels":
                self.count_files(self.train_labels, "txt"),

            "valid_images":
                self.count_files(self.valid_images, "jpg"),

            "valid_labels":
                self.count_files(self.valid_labels, "txt"),

            "test_images":
                self.count_files(self.test_images, "jpg"),

            "test_labels":
                self.count_files(self.test_labels, "txt")

        }

        LOGGER.info("Dataset statistics:")

        for k, v in stats.items():

            LOGGER.info(f"{k}: {v}")

        return stats

    ####################################################################
    # CREATE DATA YAML
    ####################################################################

    def create_yaml(

        self,

        save_path: str,

        class_names: list[str]

    ) -> None:

        save_path = Path(save_path)

        yaml = f"""path: {self.dataset_dir}

train: train/images

val: valid/images

test: test/images

nc: {len(class_names)}

names: {class_names}
"""

        save_path.write_text(yaml)

        LOGGER.info(f"Saved YAML to {save_path}")

    ####################################################################
    # REMOVE CACHE
    ####################################################################

    def remove_cache(self):

        cache = self.dataset_dir / ".cache"

        if cache.exists():

            shutil.rmtree(cache)

            LOGGER.info("Cache removed.")

    ####################################################################
    # FULL PREPARATION
    ####################################################################

    def prepare(

        self,

        yaml_path: str,

        classes: list[str]

    ) -> None:

        LOGGER.info("Preparing dataset...")

        if not self.validate_structure():

            raise RuntimeError("Dataset structure invalid.")

        self.remove_cache()

        self.dataset_statistics()

        self.create_yaml(

            yaml_path,

            classes

        )

        LOGGER.info("Dataset is ready.")