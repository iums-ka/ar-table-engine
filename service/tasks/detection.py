import os
from service.utils.transform_utils import dist_to_map
from service.utils.file_utils import load_config
from service.vision.camera import init_video_capture, preprocess_img
from service.vision.aruco import ArucoMarkerDetector, Marker
from service.ws.server import WebSocketServer

import cv2 as cv
import numpy as np


def marker_payload(marker_id, x, y):
    return {
            "Id": int(marker_id),
            "MessageType": "CONTROLHOVER",
            "Data": {
                "X": float(x),
                "Y": float(y),
            },
        }

def markers_payload(markers):
    return {
        "markers": [marker_payload(m.id, m.center.X, m.center.Y) for m in markers]
    }


def run_service(detector, cap, ws, H, preprocess=False, grace_frames=3):
    try:
        WINDOW_NAME = "MAIN"
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

        # Stable state - temporal smoothing to avoid jitter
        active_markers = {}   # id -> Marker
        miss_counts = {}      # id -> int

        while True:
            ret, frame = cap.read()

            if not ret:
                print("No frame read")
                break

            # Undistortion
            frame = cv.remap(
                frame,
                MAP_A,
                MAP_B,
                interpolation=cv.INTER_LINEAR
            )

            detection_frame = preprocess_img(frame) if preprocess else frame

            # ---- Detection ----
            detected_markers = {}  # id -> Marker (this frame only)

            corners, ids = detector.detect(detection_frame)
            if corners is not None and ids is not None:
                corners_tf = [cv.perspectiveTransform(c, H) for c in corners]
                markers = Marker.from_cv_collection(ids, corners_tf)

                for m in markers:
                    detected_markers[m.id] = m

                frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)

            # ---- Update stable state ----

            # 1. Handle detected markers (reset miss counter / add new)
            for marker_id, marker in detected_markers.items():
                active_markers[marker_id] = marker
                miss_counts[marker_id] = 0

            # 2. Handle missing markers
            for marker_id in list(active_markers.keys()):
                if marker_id not in detected_markers:
                    miss_counts[marker_id] = miss_counts.get(marker_id, 0) + 1

                    if miss_counts[marker_id] >= grace_frames:
                        del active_markers[marker_id]
                        del miss_counts[marker_id]

            # ---- Broadcast ONLY stable markers ----
            ws.broadcast(markers_payload(list(active_markers.values())))

            # ---- Debug view ----
            cv.imshow(WINDOW_NAME, cv.resize(frame, None, fx=0.5, fy=0.5))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Shutting down...")
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    CFG = load_config(r"service/config.json")
    # Load calibration
    ud = np.load(os.path.join('service/calibration', 'undistortion_args.npz'))
    camMtx = ud["camMtx"]
    distCoeffs = ud["distCoeff"]
    camMtxNew = ud["camMtxNew"]
    
    # TODO: add to camera class
    MAP_A, MAP_B = dist_to_map(camMtx,
                               distCoeffs,
                               camMtxNew,
                               CFG["camera"]["width"],
                               CFG["camera"]["height"])  

    
    detector = ArucoMarkerDetector(CFG["aruco_detection"]["physical_marker_dict"],
                                   CFG["aruco_detection"]["detector_parameters"])

    # Init camera
    cap = init_video_capture(CFG["camera"]["index"],
                             CFG["camera"]["width"],
                             CFG["camera"]["height"],
                             CFG["camera"]["fps"])

    CALIBRATION_DIR = 'service/calibration'
    BOUNDING_BOX_H = np.load(os.path.join(CALIBRATION_DIR, 'bounding_box_H.npy'))
    CAM_TO_PROJ_H = np.load(os.path.join(CALIBRATION_DIR, 'cam_to_proj_H.npy'))

    # This homopgrahpy assumes that any image displayed on the projector has been transformed
    # using the bounding box homography.
    H = CAM_TO_PROJ_H

    # Init websocket
    ws = WebSocketServer(port=5001)
    ws.start()
    
    run_service(detector, cap, ws, H, True, grace_frames=3)
    
