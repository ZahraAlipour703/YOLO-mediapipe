"""
main.py
=======

Main entry point for the YOLO + MediaPipe Hand Tracking project.

Pipeline
--------
Video
    ↓
Preprocessing
    ↓
YOLO Detection
    ↓
Object Tracking
    ↓
MediaPipe Hand Landmarks
    ↓
ArUco Pose Estimation
    ↓
Coordinate System
    ↓
Quaternion Estimation
    ↓
Joint Angle Calculation
    ↓
Visualization
    ↓
Output Video + CSV

Author
------
Zahra Alipour
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.detection.detector import YOLODetector
from src.geometry.angle_calculator import AngleCalculator
from src.geometry.coordinate_system import CoordinateSystem
from src.geometry.quaternion import QuaternionEstimator
from src.preprocessing.frame_processor import FrameProcessor
from src.preprocessing.video import VideoReader, VideoWriter
from src.tracking.aruco_tracker import ArucoTracker
from src.tracking.mediapipe_tracker import MediaPipeTracker
from src.tracking.tracker import ObjectTracker
from src.utils.config import cfg
from src.utils.logger import get_logger
from src.visualization.visualization import Visualizer


logger = get_logger("Main")


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="YOLO + MediaPipe Hand Tracking"
    )

    parser.add_argument(
        "--source",
        type=str,
        default=cfg.input_path,
        help="Video path or camera index",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output video",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():

    args = parse_args()

    logger.info("Initializing modules...")

    detector = YOLODetector()

    tracker = ObjectTracker()

    mediapipe = MediaPipeTracker()

    aruco = ArucoTracker()

    coordinate = CoordinateSystem()

    quaternion = QuaternionEstimator()

    angles = AngleCalculator()

    visualizer = Visualizer()

    processor = FrameProcessor()

    reader = VideoReader(args.source)

    writer = None

    fps = 0

    previous_time = time.time()

    for frame in reader.frames():

        frame = processor.preprocess(frame)

        # -----------------------------------------------------
        # YOLO Detection
        # -----------------------------------------------------

        detections = detector.detect(frame)

        # -----------------------------------------------------
        # Tracking
        # -----------------------------------------------------

        tracked_objects = tracker.update(detections)

        # -----------------------------------------------------
        # MediaPipe
        # -----------------------------------------------------

        hands = mediapipe.process(
            frame,
            [obj.bbox for obj in tracked_objects],
        )

        # -----------------------------------------------------
        # ArUco
        # -----------------------------------------------------

        markers = aruco.detect(frame)

        # -----------------------------------------------------
        # Coordinate Systems
        # -----------------------------------------------------

        coordinate_frames = coordinate.compute(markers)

        # -----------------------------------------------------
        # Quaternion
        # -----------------------------------------------------

        quaternions = quaternion.compute(coordinate_frames)

        # -----------------------------------------------------
        # Joint Angles
        # -----------------------------------------------------

        joint_angles = angles.compute(hands)

        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------

        output = visualizer.draw(
            frame=frame,
            detections=tracked_objects,
            hands=hands,
            markers=markers,
            coordinate_frames=coordinate_frames,
            quaternions=quaternions,
            joint_angles=joint_angles,
        )

        # -----------------------------------------------------
        # FPS
        # -----------------------------------------------------

        current = time.time()

        fps = 1.0 / (current - previous_time)

        previous_time = current

        cv2.putText(
            output,
            f"FPS: {fps:.2f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # -----------------------------------------------------
        # Video Writer
        # -----------------------------------------------------

        if args.save:

            if writer is None:

                h, w = output.shape[:2]

                writer = VideoWriter(
                    output_path=cfg.output_path,
                    fps=30,
                    frame_size=(w, h),
                )

            writer.write(output)

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        cv2.imshow(
            "YOLO + MediaPipe Hand Tracking",
            output,
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    logger.info("Finished.")

    reader.release()

    mediapipe.close()

    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()


# =============================================================================

if __name__ == "__main__":

    main()