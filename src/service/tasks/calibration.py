from service.utils.file_utils import load_config
from service.vision.camera import init_video_capture
from service.utils.transform_utils import dist_to_map
from service.vision.aruco import ArucoMarkerDetector
import time
import cv2 as cv
import numpy as np
import os


def _detect_markers_with_attempts(
        detector: ArucoMarkerDetector,
        expected_count: int = None) -> tuple:
    """
    @public
    Try capturing markers up to MAX_NR_OF_ATTEMPTS times.
    Return detected corners and ids; exit if markers cannot be detected reliably.
    """
    
    for attempt in range(1, MAX_NR_OF_ATTEMPTS + 1):
        frame = _get_most_recent_frame()
        if frame is None or np.mean(frame) < 5:
            print("Captured image is black, skipping frame.")
            continue
        
        corners, ids = detector.detect(frame, DEBUG)
        if ids is not None:
            num_ids = len(ids)
            if expected_count:
                if num_ids > expected_count:
                    print(
                        f"Detected {num_ids} markers, expected {expected_count}. Assume setup or settings are wrong; exiting."
                    )
                    os._exit(0)
                elif num_ids == expected_count:
                    print(f"Detected {expected_count} markers on attempt {attempt}.")
                    return corners, ids
            else:
                return corners, ids

    print(
        f"Failed to detect expected markers in {MAX_NR_OF_ATTEMPTS} attempts. Assume setup or settings are wrong; exiting."
    )
    os._exit(0)


def _extract_common_corners(
    corners_a: np.ndarray,
    ids_a: np.ndarray,
    corners_b: np.ndarray,
    ids_b: np.ndarray,
) -> tuple:
    """
    From detected corners and ids in both images, extract only the corners corresponding 
    to markers detected in both images. Returns the common corners ordered by marker id.
    
    Returns:
        common_grid_corners: np.ndarray of shape (N*4, 2)
        common_proj_corners: np.ndarray of shape (N*4, 2)
        common_ids: list of int
    """
    if ids_a is None or ids_b is None:
        print("No markers detected in one of the images. Exiting.")
        os._exit(0)

    # Flatten and ensure integer type
    ids_a = np.asarray(ids_a).flatten().astype(int)
    proj_ids = np.asarray(ids_b).flatten().astype(int)

    # Build lookup dictionaries
    grid_map = {id_: np.squeeze(c).astype(np.float32) for id_, c in zip(ids_a, corners_a)}
    proj_map = {id_: np.squeeze(c).astype(np.float32) for id_, c in zip(proj_ids, corners_b)}

    # Find intersection of detected IDs
    common_ids = sorted(set(ids_a).intersection(proj_ids))

    if not common_ids:
        print("No common markers detected between projected and captured images. Exiting.")
        os._exit(0)

    # Collect corresponding corners in the same sorted order
    common_grid_corners = np.concatenate([grid_map[id_].reshape(-1, 2) for id_ in common_ids], axis=0)
    common_proj_corners = np.concatenate([proj_map[id_].reshape(-1, 2) for id_ in common_ids], axis=0)

    if DEBUG:
        print(f"{len(common_ids)} common markers detected: {common_ids}")
        
    return common_grid_corners, common_proj_corners, common_ids
    

def _setup_aruco_grid():
    """
    @public
    Generate an ArUco grid board image to be projected for calibration.
    """
    rows = 9
    aruco_grid_dim = (int(rows / CFG["projector"]["height"] * CFG["projector"]["width"]), rows)
    aruco_grid = cv.aruco.GridBoard(
        aruco_grid_dim, 0.02, 0.01,
        cv.aruco.getPredefinedDictionary(getattr(cv.aruco, CFG["aruco_detection"]["projected_marker_dict"]))
    ).generateImage((CFG["projector"]["width"], CFG["projector"]["height"]), 1, 1)
    return aruco_grid


def _calibrate_camera_to_projector(flip_projector_M):
    """
    @public
    Project an ArUco grid and detect it with the camera.
    Compute the camera-to-projector homography.
    # """
    aruco_grid = _setup_aruco_grid()    
    aruco_corners_canonical, aruco_ids_canonical = DETECTOR_PROJ.detect(aruco_grid)
    
    aruco_grid_flipped = cv.warpPerspective(
                aruco_grid,
                flip_projector_M,
                (CFG["projector"]["width"], CFG["projector"]["height"])
            )

    cv.imshow(WNAME, aruco_grid_flipped)
    cv.waitKey(1)
    time.sleep(1)

    aruco_corners_camera, aruco_ids_camera = _detect_markers_with_attempts(DETECTOR_PROJ)
    
    # Convert to projector coordinate system (potentially flipped canonical coordinate frame)
    # corners: list of arrays of shape (1, 4, 2)
    aruco_corners_projector = []
    for marker in aruco_corners_canonical:  
        pts_flipped = cv.perspectiveTransform(marker, flip_projector_M)
        aruco_corners_projector.append(pts_flipped)
    aruco_corners_projector = tuple(aruco_corners_projector)
    aruco_ids_projector = aruco_ids_canonical  # For sake of completeness

    common_corners_projector, common_corners_camera, _ = _extract_common_corners(
        aruco_corners_projector, aruco_ids_projector, aruco_corners_camera, aruco_ids_camera
    )

    camera_to_projector_H, _ = cv.findHomography(np.array(common_corners_camera), np.array(common_corners_projector))
    return camera_to_projector_H


def _draw_aruco_correspondences(img_a, aruco_corners_a, img_b, aruco_corners_b):
    # aruco_corners expects shape (n, 2), where n are corners of N markers (flattened)
    
    # Ensure 3-channel images
    if len(img_a.shape) == 2:
        img_a = cv.cvtColor(img_a, cv.COLOR_GRAY2BGR)
    if len(img_b.shape) == 2:
        img_b = cv.cvtColor(img_b, cv.COLOR_GRAY2BGR)

    # Stack images horizontally
    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    canvas_h = max(h1, h2)
    canvas_w = w1 + w2
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_a
    canvas[:h2, w1:w1+w2] = img_b

    # Color scheme for corners 0,1,2,3 (repeated for each marker)
    corner_colors = [
        (0,0,255),    # red
        (0,255,0),    # green
        (255,0,0),    # blue
        (0,255,255)   # yellow
    ]

    num_corners = aruco_corners_a.shape[0]

    for i in range(num_corners):
        pt_gt = aruco_corners_a[i]
        pt_cam = aruco_corners_b[i]

        color = corner_colors[i % 4]  # repeat every 4 corners

        pt1 = (int(pt_gt[0]), int(pt_gt[1]))
        pt2 = (int(pt_cam[0] + w1), int(pt_cam[1]))  # offset camera image horizontally

        # Draw circles at corners
        cv.circle(canvas, pt1, 5, color, -1)
        cv.circle(canvas, pt2, 5, color, -1)

        # Draw connecting line
        cv.line(canvas, pt1, pt2, color, thickness=2)

    return canvas


def _get_outermost_corners(markers: np.ndarray) -> np.ndarray:
    """
    @public
    Identify the outermost corner of each marker relative to the average center,
    then order those four corners clockwise starting from the top-left.

    Returns
    -------
    np.ndarray
        Shape (1, 4, 2); ordered as: [top-left, top-right, bottom-right, bottom-left]
    """
    markers = np.asarray(markers, dtype=np.float32)
    if markers.ndim == 3 and markers.shape[1:] == (4, 2):
        markers = markers.reshape(-1, 1, 4, 2)
    elif markers.ndim == 4 and markers.shape[1:] == (1, 4, 2):
        pass
    else:
        markers = markers.reshape(-1, 1, 4, 2)

    all_corners = markers[:, 0, :, :].reshape(-1, 2)
    center = np.mean(all_corners, axis=0)

    outermost = []
    for marker in markers:
        corners = marker[0]
        dists = np.linalg.norm(corners - center, axis=1)
        outermost_idx = int(np.argmax(dists))
        outermost.append(corners[outermost_idx])

    outermost = np.asarray(outermost, dtype=np.float32)
    centroid = np.mean(outermost, axis=0)
    angles = np.arctan2(outermost[:, 1] - centroid[1], outermost[:, 0] - centroid[0])
    order = np.argsort(angles)
    ordered = outermost[order]

    top_left_index = np.argmin(np.sum(ordered, axis=1))
    ordered = np.roll(ordered, -top_left_index, axis=0)

    return ordered.reshape(1, 4, 2).astype(np.float32)


def _calibrate_bounding_box(camera_to_projector_H: np.ndarray, flip_M) -> tuple:
    """
    @public
    Detect physical ArUco markers at table corners and compute bounding box homography.
    """
    cv.imshow(WNAME, WHITE_IMG)
    cv.waitKey(1)   
    time.sleep(1) # Ensure white image is being displayed

    aruco_corners_camera, _ = _detect_markers_with_attempts(DETECTOR_PHYS, expected_count=4)

    outermost_corners_camera = _get_outermost_corners(aruco_corners_camera)
    dst_pts_projector = cv.perspectiveTransform(outermost_corners_camera, camera_to_projector_H)

    # Height and width minus 1, as the camera_to_projector_H was also computed in pixel coordinates 
    w = CFG["projector"]["width"]
    h = CFG["projector"]["height"]
    src_pts_canonical = np.asarray([[
        [0,0],
        [w-1,0],
        [w-1,h-1],
        [0,h-1]
    ]], np.float32)

    src_pts_projector = cv.perspectiveTransform(src_pts_canonical, flip_M)

    bounding_box_H, _ = cv.findHomography(src_pts_projector, dst_pts_projector)
    return bounding_box_H


def _build_flip_matrix(width, height, flip_h: bool, flip_v: bool):
    M = np.eye(3, dtype=np.float32)

    if flip_h:
        Mh = np.array([
            [-1, 0, width - 1],
            [ 0, 1, 0],
            [ 0, 0, 1]
        ], dtype=np.float32)
        M = Mh @ M

    if flip_v:
        Mv = np.array([
            [1,  0, 0],
            [0, -1, height - 1],
            [0,  0, 1]
        ], dtype=np.float32)
        M = Mv @ M

    return M


def _get_most_recent_frame():
    # Cant rely on setting buffersize to 1, so we have to do this s***
    cap = init_video_capture(CFG["camera"]["index"],
                             CFG["camera"]["width"],
                             CFG["camera"]["height"],
                             CFG["camera"]["fps"])
    _, frame = cap.read()
    cap.release()
    frame = cv.remap(
                    frame.copy(),
                    MAP_A,
                    MAP_B,
                    interpolation=cv.INTER_LINEAR
                )
    return frame


if __name__ == "__main__":
    DEBUG = True
    CFG = load_config(r"./data/config.json")

    # Load intrinsics
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

    MAX_NR_OF_ATTEMPTS = 10
    WNAME = "MAIN"
    WINDOW = cv.namedWindow(WNAME, cv.WINDOW_NORMAL)
    cv.moveWindow(WNAME, 
                    x=CFG["projector"]["screen_position"][0],
                    y=CFG["projector"]["screen_position"][1])
    cv.setWindowProperty(WNAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    DETECTOR_PHYS = ArucoMarkerDetector(CFG["aruco_detection"]["physical_marker_dict"], 
                                        CFG["aruco_detection"]["detector_parameters"])
    DETECTOR_PROJ = ArucoMarkerDetector(CFG["aruco_detection"]["projected_marker_dict"],
                                        CFG["aruco_detection"]["detector_parameters"])
    
    WHITE_IMG = np.ones((CFG["projector"]["height"], CFG["projector"]["width"], 3), np.uint8) * 255

    # front, rear, rear_upsidedown, front_upsidedown
    # To respect mirrored projector setups for IT-Trans: flip grid -> detect markers -> unflip marker-coords 
    # Build flip homography
    flip_projector_M = _build_flip_matrix(CFG["projector"]["width"], 
                                CFG["projector"]["height"], 
                                flip_h=CFG["flip"]["horizontal"], 
                                flip_v=CFG["flip"]["vertical"])

    camera_to_projector_H = _calibrate_camera_to_projector(flip_projector_M)
    bounding_box_H = _calibrate_bounding_box(camera_to_projector_H, flip_projector_M)

    print("Calibration was successful.")

    # Save calibration
    CALIBRATION_DIR = r"./data/calibration"
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    np.save(os.path.join(CALIBRATION_DIR, 'cam_to_proj_H.npy'), camera_to_projector_H)
    np.save(os.path.join(CALIBRATION_DIR, 'bounding_box_H.npy'), bounding_box_H)
    print(f"Calibration was saved to {CALIBRATION_DIR}.")


    if DEBUG:
        print("Displaying calibration bounding box in white.")
        debug_img = cv.warpPerspective(WHITE_IMG, bounding_box_H, (CFG["projector"]["width"], CFG["projector"]["height"]))
        cv.imshow(WNAME, debug_img)
        cv.waitKey(0)

