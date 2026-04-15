import os
from service.utils.transform_utils import dist_to_map
from service.utils.file_utils import load_config
from service.vision.camera import init_video_capture, preprocess_img
from service.vision.aruco.detection import Marker, ArucoMarkerDetector
from service.vision.aruco.tracking import MarkerTracker
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


def run_service(detector, marker_tracker, cap, ws, H, preprocess=False):
    try:
        WINDOW_NAME = "MAIN"
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)

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

            # Detection
            markers = []
            corners, ids = detector.detect(detection_frame)
            if corners is not None and ids is not None:
                corners_tf = [cv.perspectiveTransform(c, H) for c in corners]
                markers = Marker.from_cv_collection(ids, corners_tf)
                frame = cv.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Broadcast tracked markers
            marker_tracker.update(markers)
            ws.broadcast(markers_payload(marker_tracker.get_tracked_markers()))

            # Debug view
            cv.imshow(WINDOW_NAME, cv.resize(frame, None, fx=0.25, fy=0.25))

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Shutting down...")
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    CFG = load_config(r"./data/config.json")
    # Load calibration
    intrinsics = np.load(os.path.join('./data/calibration', 'undistortion_args.npz'))
    camMtx = intrinsics["camMtx"]
    distCoeffs = intrinsics["distCoeff"]
    camMtxNew = intrinsics["camMtxNew"]
    
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


    grace_frames = 5
    min_observations = 5
    marker_tracker = MarkerTracker(grace_frames, min_observations) 

    CALIBRATION_DIR = r"./data/calibration"
    BOUNDING_BOX_H = np.load(os.path.join(CALIBRATION_DIR, 'bounding_box_H.npy'))
    CAM_TO_PROJ_H = np.load(os.path.join(CALIBRATION_DIR, 'cam_to_proj_H.npy'))

    # This homopgrahpy assumes that any image displayed on the projector has been transformed
    # using the bounding box homography.
    H = CAM_TO_PROJ_H

    # Init websocket
    ws = WebSocketServer(port=5001)
    ws.start()
    
    run_service(detector, marker_tracker, cap, ws, H, True)
    
