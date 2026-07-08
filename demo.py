"""
demo.py
=======

Demo script for the YOLO + MediaPipe Hand Tracking project.

Runs the complete pipeline:
    Video/Image
        ↓
    YOLO Detection
        ↓
    Hand Tracking
        ↓
    MediaPipe Landmarks
        ↓
    Angle Estimation
        ↓
    Visualization

Author
------
Zahra Alipour
"""

from pathlib import Path
import argparse
import cv2

from src.detector import YOLODetector
from src.mediapipe_tracker import MediaPipeTracker
from src.tracker import HandTracker
from src.angle_calculator import AngleCalculator
from src.visualization import Visualizer
from src.config import cfg


# ============================================================
# Demo Runner
# ============================================================

def run_demo(source: str) -> None:
    """
    Run the complete hand tracking demo.

    Parameters
    ----------
    source : str
        Image path, video path, or webcam index ("0").
    """

    detector = YOLODetector(
        model_path=cfg.model_path,
        confidence=cfg.confidence,
        iou=cfg.iou,
    )

    tracker = HandTracker()

    mp_tracker = MediaPipeTracker(
        max_num_hands=cfg.max_hands,
        detection_confidence=cfg.detection_confidence,
        tracking_confidence=cfg.tracking_confidence,
    )

    angle_calculator = AngleCalculator()

    visualizer = Visualizer()

    # ------------------------------------------------------

    if source == "0":
        cap = cv2.VideoCapture(0)

    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # ------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -----------------------------------------------
        # YOLO Detection
        # -----------------------------------------------

        detections = detector.detect(frame)

        # -----------------------------------------------
        # Tracking
        # -----------------------------------------------

        tracks = tracker.update(detections)

        # -----------------------------------------------
        # MediaPipe
        # -----------------------------------------------

        boxes = [track.bbox for track in tracks]

        hands = mp_tracker.process(frame, boxes)

        # -----------------------------------------------
        # Angle Calculation
        # -----------------------------------------------

        for hand in hands:

            hand.angles = angle_calculator.calculate(hand.landmarks)

        # -----------------------------------------------
        # Draw Results
        # -----------------------------------------------

        frame = visualizer.draw(frame, tracks, hands)

        cv2.imshow("YOLO + MediaPipe Hand Tracking", frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video path, image path or webcam (0).",
    )

    return parser.parse_args()


def main() -> None:
    """
    Program entry.
    """

    args = parse_args()

    run_demo(args.source)


if __name__ == "__main__":
    main()