"""
segmentation.py
===============

Segment Anything (SAM) Hand Segmentation

This module performs hand segmentation using Meta's Segment Anything Model.
YOLO bounding boxes are used as prompts to SAM.

Pipeline
--------
Frame
   │
YOLO Detection
   │
Bounding Box
   │
SAM Predictor
   │
Binary Mask
   │
Overlay / Cropped Hand

Author:
    Zahra Alipour
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


class HandSegmenter:
    """
    Segment hands using SAM.

    Parameters
    ----------
    model_type : str
        SAM backbone (vit_b, vit_l, vit_h)

    checkpoint : str
        Path to SAM checkpoint.

    device : str
        cpu or cuda
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cpu",
    ):

        checkpoint = Path(checkpoint)

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        sam = sam_model_registry[model_type](
            checkpoint=str(checkpoint)
        )

        sam.to(device)

        self.predictor = SamPredictor(sam)

    # ----------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        boxes: List[List[int]],
    ) -> List[np.ndarray]:
        """
        Generate masks for YOLO detections.

        Parameters
        ----------
        image
            BGR frame

        boxes
            List of YOLO bounding boxes

        Returns
        -------
        list[np.ndarray]
            Binary masks
        """

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(rgb)

        masks = []

        for box in boxes:

            box = np.array(box)

            mask, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
            )

            masks.append(mask[0])

        return masks

    # ----------------------------------------------------------

    @staticmethod
    def overlay(
        image: np.ndarray,
        masks: List[np.ndarray],
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay masks on image.
        """

        output = image.copy()

        for mask in masks:

            color = np.random.randint(
                0,
                255,
                size=3,
                dtype=np.uint8,
            )

            colored = np.zeros_like(image)

            colored[mask] = color

            output = cv2.addWeighted(
                output,
                1.0,
                colored,
                alpha,
                0,
            )

        return output

    # ----------------------------------------------------------

    @staticmethod
    def crop_segments(
        image: np.ndarray,
        masks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Extract segmented hand images.
        """

        segments = []

        for mask in masks:

            result = np.zeros_like(image)

            result[mask] = image[mask]

            segments.append(result)

        return segments

    # ----------------------------------------------------------

    @staticmethod
    def save_masks(
        masks: List[np.ndarray],
        output_dir: str,
    ) -> None:
        """
        Save binary masks.
        """

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        for i, mask in enumerate(masks):

            cv2.imwrite(
                str(output / f"mask_{i}.png"),
                mask.astype(np.uint8) * 255,
            )