"""
tracker.py
==========

High-level tracking pipeline.

This module combines:

1. YOLO Hand Detection
2. MediaPipe Landmark Tracking
3. ArUco Marker Pose Estimation
4. Finger Joint Angle Calculation

Author:
    Zahra Alipour
"""

from __future__ import annotations

from typing import Dict, Any

import cv2
import numpy as np

from src.detection.detector import YOLODetector
from src.tracking.mediapipe_tracker import MediaPipeTracker
from src.tracking.aruco_tracker import ArucoTracker
from src.geometry.angle_calculator import AngleCalculator


class HandTracker:
    """
    Complete hand tracking pipeline.

    Workflow
    --------
    Frame
      │
      ▼
    YOLO Detection
      │
      ▼
    MediaPipe Landmarks
      │
      ▼
    ArUco Pose Estimation
      │
      ▼
    Joint Angle Calculation
      │
      ▼
    Results
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
    ) -> None:

        self.detector = YOLODetector(
            model_path=model_path,
            confidence=confidence,
        )

        self.landmark_tracker = MediaPipeTracker()

        self.aruco_tracker = ArucoTracker()

        self.angle_calculator = AngleCalculator()

    # -------------------------------------------------------------

    def process(
        self,
        frame: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Parameters
        ----------
        frame : ndarray
            Input BGR frame.

        Returns
        -------
        dict
            Detection results.
        """

        # -------------------------------------------------
        # Step 1 : Detect hands
        # -------------------------------------------------

        detections = self.detector.detect(frame)

        boxes = [det.bbox for det in detections]

        # -------------------------------------------------
        # Step 2 : Estimate landmarks
        # -------------------------------------------------

        hands = self.landmark_tracker.process(
            frame,
            boxes,
        )

        # -------------------------------------------------
        # Step 3 : Detect ArUco markers
        # -------------------------------------------------

        markers = self.aruco_tracker.process(frame)

        # -------------------------------------------------
        # Step 4 : Compute joint angles
        # -------------------------------------------------

        angles = []

        for hand in hands:

            angle = self.angle_calculator.compute(
                hand.landmarks
            )

            angles.append(angle)

        # -------------------------------------------------

        return {
            "detections": detections,
            "hands": hands,
            "markers": markers,
            "angles": angles,
        }

    # -------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        results: Dict[str, Any],
    ) -> np.ndarray:
        """
        Draw all tracking results.
        """

        image = frame.copy()

        image = self.detector.draw(
            image,
            results["detections"],
        )

        image = self.landmark_tracker.draw(
            image,
            results["hands"],
        )

        image = self.aruco_tracker.draw(
            image,
            results["markers"],
        )

        image = self.angle_calculator.draw(
            image,
            results["hands"],
            results["angles"],
        )

        return image

    # -------------------------------------------------------------

    def close(self) -> None:
        """
        Release all resources.
        """

        self.landmark_tracker.close()