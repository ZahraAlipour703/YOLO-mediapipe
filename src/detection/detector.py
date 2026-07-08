"""
detector.py
===========

YOLOv8 Hand Detector

This module loads a trained YOLOv8 model and performs hand detection on
images or video frames.

Author: Zahra Alipour
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class HandDetector:
    """
    Wrapper around Ultralytics YOLO model.

    Example
    -------
    detector = HandDetector("models/custom_model.pt")

    boxes, scores = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        device: str = "cpu",
    ) -> None:

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.confidence = confidence
        self.device = device

        self.model = YOLO(str(self.model_path))

    # -------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[List[int]], List[float]]:
        """
        Detect hands.

        Parameters
        ----------
        frame
            OpenCV image (BGR)

        Returns
        -------
        boxes
            List of bounding boxes

        scores
            Confidence scores
        """

        prediction = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        boxes = []
        scores = []

        if len(prediction) == 0:
            return boxes, scores

        result = prediction[0]

        if result.boxes is None:
            return boxes, scores

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy.cpu().numpy()[0].astype(int)
            )

            conf = float(box.conf.cpu())

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        return boxes, scores

    # -------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        boxes: List[List[int]],
        scores: List[float],
    ) -> np.ndarray:
        """
        Draw detections on frame.
        """

        image = frame.copy()

        for box, score in zip(boxes, scores):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                f"{score:.2f}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return image

    # -------------------------------------------------------------

    def detect_and_draw(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience function.

        Detect hands and draw bounding boxes.
        """

        boxes, scores = self.detect(frame)

        return self.draw(frame, boxes, scores)