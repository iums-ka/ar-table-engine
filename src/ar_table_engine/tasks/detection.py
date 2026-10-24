import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from ar_table_engine.utils.drawing import draw_marker_orientations
from ar_table_engine.utils.file_utils import load_config
from ar_table_engine.utils.transform_utils import dist_to_map
from ar_table_engine.utils.window import Window
from ar_table_engine.vision.aruco.detection import MarkerDetector
from ar_table_engine.vision.aruco.marker import Marker, markers_from_arrays
from ar_table_engine.vision.aruco.tracking import MarkerTracker
from ar_table_engine.vision.camera import init_video_capture, preprocess_img
from ar_table_engine.ws.server import WebSocketServer


@dataclass
class MarkerPayload:
    id: int
    x: float
    y: float
    orientation: tuple[float, float]


def markers_payload(markers: list[Marker]):
    return {
        "markers": [
            asdict(
                MarkerPayload(
                    m.id,
                    float(m.center[0]),
                    float(m.center[1]),
                    (float(m.orientation[0]), float(m.orientation[1])),
                )
            )
            for m in markers
        ]
    }


def run_service(
    detector: MarkerDetector,
    marker_tracker: MarkerTracker,
    cap: cv.VideoCapture,
    ws: WebSocketServer,
    H: np.ndarray,
    undistort=True,
    preprocess=False,
):
    window = Window("Detection-Debug-Window")

    fps = 0.0
    previous_frame_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("No frame read")
                break

            # Undistortion
            if undistort:
                frame = cv.remap(frame, MAP_A, MAP_B, interpolation=cv.INTER_LINEAR)

            detection_frame = preprocess_img(frame) if preprocess else frame

            # Detection: measure only the detector call
            detect_start = time.perf_counter()
            corners, ids = detector.detect(detection_frame)
            # print(corners)
            detect_end = time.perf_counter()

            detect_ms = (detect_end - detect_start) * 1000.0

            # Detection results
            markers = []
            markers_tf = []
            marker_count = 0

            if corners is not None and ids is not None:
                marker_count = len(ids)

                corners_tf = [cv.perspectiveTransform(c, H) for c in corners]

                markers = markers_from_arrays(corners, ids)
                markers_tf = markers_from_arrays(corners_tf, ids)

                frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)

            # Broadcast tracked markers
            marker_tracker.update(markers_tf)
            stable = marker_tracker.get_stable_markers()

            payload = markers_payload(stable)
            ws.broadcast(payload)

            # Draw marker orientation
            draw_marker_orientations(frame, markers, 100)

            # Estimate end-to-end loop FPS, smoothed like the C++ version
            now = time.perf_counter()
            dt = now - previous_frame_time
            previous_frame_time = now

            if dt > 0.0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            # Debug overlay
            debug_text = (
                f"Nano: {marker_count} markers"
                f" | detect: {detect_ms:.1f} ms"
                f" | FPS: {fps:.1f}"
            )

            cv.putText(
                frame,
                debug_text,
                (30, 50),
                cv.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )

            # Debug view
            window.show(frame)

            if cv.pollKey() & 0xFF == ord("q"):
                break

    finally:
        print("Shutting down...")
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    CFG = load_config(r"./data/config.json")
    # Load calibration
    intrinsics_path = Path(CFG["intrinsics_path"])
    intrinsics = np.load(intrinsics_path)

    camMtx = intrinsics["camMtx"]
    distCoeffs = intrinsics["distCoeff"]
    camMtxNew = intrinsics["camMtxNew"]

    print(camMtx, distCoeffs, camMtxNew)
    # exit()
    # TODO: add to camera class
    MAP_A, MAP_B = dist_to_map(
        camMtx, distCoeffs, camMtxNew, CFG["camera"]["width"], CFG["camera"]["height"]
    )

    detector = MarkerDetector(
        CFG["aruco_detection"]["physical_marker_dict"],
        CFG["aruco_detection"].get("detector_parameters"),
    )

    # Init camera
    cap = init_video_capture(
        CFG["camera"]["index"],
        CFG["camera"]["width"],
        CFG["camera"]["height"],
        CFG["camera"]["fps"],
    )

    undistort = CFG["undistort"]
    preprocess = CFG["preprocess"]

    grace_frames = 5
    min_observations = 5
    marker_tracker = MarkerTracker(grace_frames, min_observations)

    CALIBRATION_DIR = r"./data/calibration"
    BOUNDING_BOX_H = np.load(os.path.join(CALIBRATION_DIR, "bounding_box_H.npy"))
    CAM_TO_PROJ_H = np.load(os.path.join(CALIBRATION_DIR, "cam_to_proj_H.npy"))

    # Currently we do not use the bounding box homography as we calibrate the bounding box / projection area using the projector itself (frontend agnostic)
    H = CAM_TO_PROJ_H

    # Init websocket
    ws = WebSocketServer(port=5001)
    ws.start()

    run_service(detector, marker_tracker, cap, ws, H, undistort, preprocess)
