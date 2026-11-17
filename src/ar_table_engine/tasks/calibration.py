import os
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from dotenv import load_dotenv

from ar_table_engine.utils.file_utils import load_config
from ar_table_engine.utils.transform_utils import dist_to_map
from ar_table_engine.vision.aruco.detection import MarkerDetector
from ar_table_engine.vision.camera import init_video_capture

# def _detect_markers_with_attempts(
#     detector: MarkerDetector, expected_count: int = None
# ) -> tuple:
#     """@public
#     Try capturing markers up to MAX_NR_OF_ATTEMPTS times.
#     Return detected corners and ids; exit if markers cannot be detected reliably.
#     """
#     for attempt in range(1, MAX_NR_OF_ATTEMPTS + 1):
#         frame = _get_most_recent_frame()
#         if frame is None or np.mean(frame) < 5:
#             print("Captured image is black, skipping frame.")
#             continue

#         corners, ids = detector.detect(frame, DEBUG)
#         if ids is not None:
#             num_ids = len(ids)
#             if expected_count:
#                 if num_ids > expected_count:
#                     print(
#                         f"Detected {num_ids} markers, expected {expected_count}. Assume setup or settings are wrong; exiting."
#                     )
#                     os._exit(0)
#                 elif num_ids == expected_count:
#                     print(f"Detected {expected_count} markers on attempt {attempt}.")
#                     return corners, ids
#             else:
#                 return corners, ids

#     print(
#         f"Failed to detect expected markers in {MAX_NR_OF_ATTEMPTS} attempts. Assume setup or settings are wrong; exiting."
#     )
#     os._exit(0)


def _detect_markers_with_attempts(
    detector: MarkerDetector,
    expected_count: int = None,
    min_count: int = None,
) -> tuple:
    """@public
    Try detecting ArUco markers up to MAX_NR_OF_ATTEMPTS times.

    Args:
        detector: ArUco marker detector.
        expected_count: Require exactly this many markers.
        min_count: Require at least this many markers.

    Returns:
        Tuple of detected corners and IDs.

    Raises:
        RuntimeError: If the required number of markers cannot be detected.
    """
    if expected_count is not None and min_count is not None:
        raise ValueError("Specify either expected_count or min_count, not both.")

    for attempt in range(1, MAX_NR_OF_ATTEMPTS + 1):
        frame = _get_most_recent_frame()

        if frame is None or np.mean(frame) < 5:
            print("Captured image is black or invalid, skipping frame.")
            continue

        corners, ids = detector.detect(frame, DEBUG)
        num_ids = len(ids) if ids is not None else 0

        if expected_count is not None:
            success = num_ids == expected_count
        elif min_count is not None:
            success = num_ids >= min_count
        else:
            success = num_ids > 0

        if success:
            print(f"Detected {num_ids} markers on attempt {attempt}.")
            return corners, ids

        print(
            f"Attempt {attempt}/{MAX_NR_OF_ATTEMPTS}: "
            f"Detected {num_ids} markers, "
            f"required {'exactly ' + str(expected_count) if expected_count is not None else 'at least ' + str(min_count) if min_count is not None else 'at least 1'}."
        )

    raise RuntimeError(
        f"Failed to detect required markers after {MAX_NR_OF_ATTEMPTS} attempts."
    )


def _extract_common_corners(
    corners_a: np.ndarray,
    ids_a: np.ndarray,
    corners_b: np.ndarray,
    ids_b: np.ndarray,
) -> tuple:
    """From detected corners and ids in both images, extract only the corners corresponding
    to markers detected in both images. Returns the common corners ordered by marker id.

    Returns:
        common_grid_corners: np.ndarray of shape (N*4, 2)
        common_proj_corners: np.ndarray of shape (N*4, 2)
        common_ids: list of int
    """
    if ids_a is None or ids_b is None:
        raise RuntimeError("No markers detected in one of the images.")

    # Flatten and ensure integer type
    ids_a = np.asarray(ids_a).flatten().astype(int)
    proj_ids = np.asarray(ids_b).flatten().astype(int)

    # Build lookup dictionaries
    grid_map = {
        id_: np.squeeze(c).astype(np.float32) for id_, c in zip(ids_a, corners_a)
    }
    proj_map = {
        id_: np.squeeze(c).astype(np.float32) for id_, c in zip(proj_ids, corners_b)
    }

    # Find intersection of detected IDs
    common_ids = sorted(set(ids_a).intersection(proj_ids))

    if len(common_ids) < 4:
        raise RuntimeError(
            f"Insufficient common ArUco markers: "
            f"detected {len(common_ids)}, required at least 4."
        )

    # Collect corresponding corners in the same sorted order
    common_grid_corners = np.concatenate(
        [grid_map[id_].reshape(-1, 2) for id_ in common_ids], axis=0
    )
    common_proj_corners = np.concatenate(
        [proj_map[id_].reshape(-1, 2) for id_ in common_ids], axis=0
    )

    if DEBUG:
        print(f"{len(common_ids)} common markers detected: {common_ids}")

    return common_grid_corners, common_proj_corners, common_ids


def _setup_aruco_grid():
    """@public
    Generate an ArUco grid board image to be projected for calibration.
    """
    rows = 9
    aruco_grid_dim = (
        int(rows / CFG["projector"]["height"] * CFG["projector"]["width"]),
        rows,
    )
    aruco_grid = cv.aruco.GridBoard(
        aruco_grid_dim,
        0.02,
        0.01,
        cv.aruco.getPredefinedDictionary(
            getattr(cv.aruco, CFG["aruco_detection"]["projected_marker_dict"])
        ),
    ).generateImage((CFG["projector"]["width"], CFG["projector"]["height"]), 1, 1)
    return aruco_grid


def _calibrate_camera_to_projector(flip_projector_M):
    """@public
    Project an ArUco grid and detect it with the camera.
    Compute the camera-to-projector homography.
    #
    """
    aruco_grid = _setup_aruco_grid()
    aruco_corners_canonical, aruco_ids_canonical = DETECTOR_PROJ.detect(aruco_grid)

    aruco_grid_flipped = cv.warpPerspective(
        aruco_grid,
        flip_projector_M,
        (CFG["projector"]["width"], CFG["projector"]["height"]),
    )

    _show_calibration_pattern(
        aruco_grid_flipped,
        "Projecting ArUco grid.",
    )

    # aruco_corners_camera, aruco_ids_camera = _detect_markers_with_attempts(
    #     DETECTOR_PROJ
    # )

    aruco_corners_camera, aruco_ids_camera = _detect_markers_with_attempts(
        DETECTOR_PROJ,
        min_count=4,
    )

    # Convert to projector coordinate system (potentially flipped canonical coordinate frame)
    # corners: list of arrays of shape (1, 4, 2)
    aruco_corners_projector = []
    for marker in aruco_corners_canonical:
        pts_flipped = cv.perspectiveTransform(marker, flip_projector_M)
        aruco_corners_projector.append(pts_flipped)
    aruco_corners_projector = tuple(aruco_corners_projector)
    aruco_ids_projector = aruco_ids_canonical  # For sake of completeness

    common_corners_projector, common_corners_camera, _ = _extract_common_corners(
        aruco_corners_projector,
        aruco_ids_projector,
        aruco_corners_camera,
        aruco_ids_camera,
    )

    # camera_to_projector_H, _ = cv.findHomography(
    #     np.array(common_corners_camera), np.array(common_corners_projector)
    # )
    # return camera_to_projector_H
    camera_to_projector_H, _ = cv.findHomography(
        common_corners_camera,
        common_corners_projector,
    )

    if camera_to_projector_H is None:
        raise RuntimeError("Camera-to-projector homography estimation failed.")

    _print_homography_metrics(
        camera_to_projector_H,
        common_corners_camera,
        common_corners_projector,
        "Camera-to-projector",
    )

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
    canvas[:h2, w1 : w1 + w2] = img_b

    # Color scheme for corners 0,1,2,3 (repeated for each marker)
    corner_colors = [
        (0, 0, 255),  # red
        (0, 255, 0),  # green
        (255, 0, 0),  # blue
        (0, 255, 255),  # yellow
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
    """@public
    Identify the outermost corner of each marker relative to the average center,
    then order those four corners clockwise starting from the top-left.

    Returns:
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
    """@public
    Detect physical ArUco markers at table corners and compute bounding box homography.
    """
    # cv.imshow(WNAME, WHITE_IMG)
    # cv.waitKey(1)
    # time.sleep(1)  # Ensure white image is being displayed

    _show_calibration_pattern(
        WHITE_IMG,
        "Projecting white image for physical marker detection.",
    )

    aruco_corners_camera, _ = _detect_markers_with_attempts(
        DETECTOR_PHYS, expected_count=4
    )

    outermost_corners_camera = _get_outermost_corners(aruco_corners_camera)
    dst_pts_projector = cv.perspectiveTransform(
        outermost_corners_camera, camera_to_projector_H
    )

    # Height and width minus 1, as the camera_to_projector_H was also computed in pixel coordinates
    w = CFG["projector"]["width"]
    h = CFG["projector"]["height"]
    src_pts_canonical = np.asarray(
        [[[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]], np.float32
    )

    src_pts_projector = cv.perspectiveTransform(src_pts_canonical, flip_M)

    # bounding_box_H, _ = cv.findHomography(src_pts_projector, dst_pts_projector)
    # return bounding_box_H
    bounding_box_H, _ = cv.findHomography(
        src_pts_projector,
        dst_pts_projector,
    )

    if bounding_box_H is None:
        raise RuntimeError("Bounding-box homography estimation failed.")

    _print_homography_metrics(
        bounding_box_H,
        src_pts_projector,
        dst_pts_projector,
        "Bounding-box",
    )

    _print_bounding_box_metrics(
        dst_pts_projector,
        w,
        h,
    )

    return bounding_box_H


def _build_flip_matrix(width, height, flip_h: bool, flip_v: bool):
    M = np.eye(3, dtype=np.float32)

    if flip_h:
        Mh = np.array([[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        M = Mh @ M

    if flip_v:
        Mv = np.array([[1, 0, 0], [0, -1, height - 1], [0, 0, 1]], dtype=np.float32)
        M = Mv @ M

    return M


# def _get_most_recent_frame():
#     # Cant rely on setting buffersize to 1, so we have to do this s***
#     cap = init_video_capture(
#         CFG["camera"]["index"],
#         CFG["camera"]["width"],
#         CFG["camera"]["height"],
#         CFG["camera"]["fps"],
#     )
#     _, frame = cap.read()
#     cap.release()
#     frame = cv.remap(frame.copy(), MAP_A, MAP_B, interpolation=cv.INTER_LINEAR)
#     return frame


def _get_most_recent_frame():
    cap = init_video_capture(
        CFG["camera"]["index"],
        CFG["camera"]["width"],
        CFG["camera"]["height"],
        CFG["camera"]["fps"],
    )

    try:
        success, frame = cap.read()
    finally:
        cap.release()

    if not success or frame is None:
        return None

    return cv.remap(
        frame,
        MAP_A,
        MAP_B,
        interpolation=cv.INTER_LINEAR,
    )


def _print_homography_metrics(H, src_pts, dst_pts, name):
    """Print reprojection metrics for a computed homography."""
    src = np.asarray(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst_pts, dtype=np.float32).reshape(-1, 2)

    projected = cv.perspectiveTransform(src, H).reshape(-1, 2)
    errors = np.linalg.norm(projected - dst, axis=1)

    rmse = np.sqrt(np.mean(errors**2))

    print(f"\n{name} calibration metrics:")
    print(f"  Correspondences: {len(errors)}")
    print(f"  Reprojection RMSE: {rmse:.3f} px")
    print(f"  Mean error: {np.mean(errors):.3f} px")
    print(f"  Max error: {np.max(errors):.3f} px")


def _print_bounding_box_metrics(corners, width, height):
    """Print geometric metrics for the detected bounding box."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    area = abs(cv.contourArea(pts))
    area_ratio = area / (width * height)

    edges = np.roll(pts, -1, axis=0) - pts
    edge_lengths = np.linalg.norm(edges, axis=1)

    print("\nBounding-box geometry metrics:")
    print(f"  Convex: {cv.isContourConvex(pts)}")
    print(f"  Area: {area:.1f} px²")
    print(f"  Area ratio: {area_ratio:.3f}")
    print(f"  Min edge length: {np.min(edge_lengths):.1f} px")
    print(f"  Max edge length: {np.max(edge_lengths):.1f} px")


def _show_calibration_pattern(image, message=None):
    """Display a calibration pattern and optionally wait for user input."""
    cv.imshow(WNAME, image)
    cv.waitKey(1)  # Ensure the image is rendered

    if DEBUG:
        if message:
            print(f"{message} Press any key to continue.")
        cv.waitKey(0)
    else:
        time.sleep(1)  # Allow projector and camera to settle


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(PROJECT_ROOT / ".env")
    CFG_PATH = os.environ.get("CONFIG_PATH")
    CFG = load_config(CFG_PATH)
    DEBUG = CFG.get("calibration_debug", False)

    # Load intrinsics
    intrinsics_path = Path(CFG["intrinsics_path"])
    intrinsics = np.load(intrinsics_path)
    camMtx = intrinsics["camMtx"]
    distCoeffs = intrinsics["distCoeff"]
    camMtxNew = intrinsics["camMtxNew"]

    # TODO: add to camera class
    MAP_A, MAP_B = dist_to_map(
        camMtx, distCoeffs, camMtxNew, CFG["camera"]["width"], CFG["camera"]["height"]
    )

    MAX_NR_OF_ATTEMPTS = 10
    WNAME = "MAIN"
    WINDOW = cv.namedWindow(WNAME, cv.WINDOW_NORMAL)
    cv.moveWindow(
        WNAME,
        x=CFG["projector"]["screen_position"][0],
        y=CFG["projector"]["screen_position"][1],
    )
    cv.setWindowProperty(WNAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    DETECTOR_PHYS = MarkerDetector(
        CFG["aruco_detection"]["physical_marker_dict"],
        CFG["aruco_detection"].get("detector_parameters"),
    )
    DETECTOR_PROJ = MarkerDetector(
        CFG["aruco_detection"]["projected_marker_dict"],
        CFG["aruco_detection"].get("detector_parameters"),
    )

    WHITE_IMG = (
        np.ones((CFG["projector"]["height"], CFG["projector"]["width"], 3), np.uint8)
        * 255
    )

    # front, rear, rear_upsidedown, front_upsidedown
    # To respect mirrored projector setups for IT-Trans: flip grid -> detect markers -> unflip marker-coords
    # Build flip homography
    flip_projector_M = _build_flip_matrix(
        CFG["projector"]["width"],
        CFG["projector"]["height"],
        flip_h=CFG["flip"]["horizontal"],
        flip_v=CFG["flip"]["vertical"],
    )

    # Create calibration directory if it doesn't exist
    CALIBRATION_DIR = r"./data/calibration"
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

    try:
        camera_to_projector_H = _calibrate_camera_to_projector(flip_projector_M)

        bounding_box_H = None

        if CFG.get("compute_bounding_box_H", False):
            bounding_box_H = _calibrate_bounding_box(
                camera_to_projector_H,
                flip_projector_M,
            )

        # Save only after all requested calibrations succeed
        np.save(
            os.path.join(CALIBRATION_DIR, "cam_to_proj_H.npy"),
            camera_to_projector_H,
        )
        print("Camera-to-projector calibration was successful.")

        if bounding_box_H is not None:
            np.save(
                os.path.join(CALIBRATION_DIR, "bounding_box_H.npy"),
                bounding_box_H,
            )
            print("Bounding-box calibration was successful.")

        print(f"Calibration saved to {CALIBRATION_DIR}.")

        if DEBUG and bounding_box_H is not None:
            print("Displaying calibration bounding box. Press any key to exit.")

            debug_img = cv.warpPerspective(
                WHITE_IMG,
                bounding_box_H,
                (CFG["projector"]["width"], CFG["projector"]["height"]),
            )

            cv.imshow(WNAME, debug_img)
            cv.waitKey(0)

    except (RuntimeError, ValueError, cv.error) as exc:
        print(f"Calibration failed: {exc}")
        raise SystemExit(1)

    finally:
        cv.destroyAllWindows()
