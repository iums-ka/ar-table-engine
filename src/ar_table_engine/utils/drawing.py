import cv2 as cv
import numpy as np

from ar_table_engine.vision.aruco.marker import Marker


def draw_marker_orientations(
    frame: np.ndarray,
    markers: list[Marker],
    vector_length: int = 100,
) -> np.ndarray:
    """Draw current, reference, and world-up orientation vectors.

    Colors (BGR):
        Blue:  reference orientation
        Red:   current orientation
        Green: world up
    """

    def endpoint(center: np.ndarray, direction: np.ndarray) -> tuple[int, int]:
        direction = np.asarray(direction, dtype=np.float32).reshape(2)
        norm = np.linalg.norm(direction)

        if norm == 0:
            return int(center[0]), int(center[1])

        direction = direction / norm
        point = center + direction * vector_length
        return int(round(point[0])), int(round(point[1]))

    for marker in markers:
        center = marker.center

        origin = (int(round(center[0])), int(round(center[1])))

        # Current orientation (red)
        cv.line(
            frame,
            origin,
            endpoint(center, marker.orientation),
            color=(0, 0, 255),
            thickness=2,
        )

    return frame
