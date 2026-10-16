import cv2 as cv
import numpy as np


class MarkerDetector:
    """Class for detecting ArUco markers in images using specific parameters."""

    def __init__(self, aruco_dict: str, detector_params: dict = None) -> None:
        self.aruco_dict = cv.aruco.getPredefinedDictionary(
            getattr(cv.aruco, aruco_dict)
        )

        # setup marker detector
        if detector_params is None:
            self.detector = cv.aruco.ArucoDetector(self.aruco_dict)
        else:
            self.detector_params = cv.aruco.DetectorParameters()
            for key, value in detector_params.items():
                setattr(self.detector_params, key, value)
            self.detector = cv.aruco.ArucoDetector(
                self.aruco_dict, self.detector_params
            )

    def detect(self, img: np.ndarray, debug: bool = False) -> tuple:
        """Detects ArUco markers in the given image and returns their corners and IDs."""
        # corners shape: N-sized tuple of (1, 4, 2) entries
        # ids shape: (N, 1)
        corners, ids, _ = self.detector.detectMarkers(img)
        if debug:
            debug_img = img.copy()
            if ids is not None:
                debug_img = cv.aruco.drawDetectedMarkers(debug_img, corners, ids)
            cv.imshow(
                "Detected ArUco Markers", cv.resize(debug_img, (0, 0), fx=0.4, fy=0.4)
            )  # TODO: adjust this to actual screen size
            print(
                "Displaying debug window. To continue, select the debug window and then press any key."
            )
            cv.waitKey(0)
            cv.destroyWindow("Detected ArUco Markers")
        return corners, ids
