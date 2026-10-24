import cv2 as cv
import numpy as np

from . import aruco_nano_ext


class MarkerDetector:
    """Detect ArUco markers using ArUco Nano."""

    def __init__(
        self,
        aruco_dict: str,
        detector_params: dict | None = None,
    ) -> None:
        dictionary_id = int(getattr(cv.aruco, aruco_dict))

        params = aruco_nano_ext.DetectorParameters()

        if detector_params is not None:
            for key, value in detector_params.items():
                setattr(params, key, value)

        self.detector = aruco_nano_ext.Detector(dictionary_id, params)

    def detect(
        self,
        img: np.ndarray,
        debug: bool = False,
    ) -> tuple:
        """Detect markers and return corners and IDs.

        Returns:
            corners: list of N arrays, each with shape (1, 4, 2),
                     matching OpenCV's ArucoDetector convention.
            ids: ndarray of shape (N, 1), or None if no markers are found.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a NumPy array")

        if img.dtype != np.uint8:
            raise ValueError("img must have dtype uint8")

        if img.ndim not in (2, 3):
            raise ValueError("img must be grayscale or a BGR color image")

        if img.ndim == 3 and img.shape[2] != 3:
            raise ValueError("Color images must have 3 BGR channels")

        img = np.ascontiguousarray(img)
        corners_array, ids_array = self.detector.detect(img)

        if len(ids_array) == 0:
            corners = []
            ids = None
        else:
            # Convert Nano's (N, 4, 2) output to OpenCV's conventions.
            corners = [corners_array[i : i + 1] for i in range(len(ids_array))]
            ids = ids_array.reshape(-1, 1)

        if debug:
            debug_img = img.copy()
            if ids is not None:
                debug_img = cv.aruco.drawDetectedMarkers(debug_img, corners, ids)

            cv.imshow(
                "Detected ArUco Markers",
                cv.resize(debug_img, (0, 0), fx=0.4, fy=0.4),
            )
            print("Displaying debug window. Select it and press any key to continue.")
            cv.waitKey(0)
            cv.destroyWindow("Detected ArUco Markers")

        return corners, ids


if __name__ == "__main__":
    print("Hello")
    print(aruco_nano_ext)
