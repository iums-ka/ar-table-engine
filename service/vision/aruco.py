import cv2 as cv
import numpy as np


class Coordinate:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y


class Marker:
    @staticmethod
    def to_cv_collection(markers):  # Input: Marker
        corners_cv = []
        ids_cv = []
        for m in markers:
            corners = np.array(
                [
                    [m.TL.X, m.TL.Y],
                    [m.TR.X, m.TR.Y],
                    [m.BR.X, m.BR.Y],
                    [m.BL.X, m.BL.Y],
                ],
                dtype=np.float32,
            ).reshape(1, 4, 2)

            corners_cv.append(corners)
            ids_cv.append([m.id])
        return corners_cv, np.array(ids_cv, dtype=np.int32)
    @staticmethod
    def from_cv_collection(ids, corners):  # Input: N-size Tuple of (1, 4, 2) - cv format
        ids_cv = ids[:, np.newaxis, :]
        return [Marker(*m) for m in zip(ids_cv, corners)]
    def to_cv(self):  # Output: (corners, id) in cv format
        return self.corners_cv, np.array([[self.id]])
    def __init__(self, id, corners):
        self.id = int(np.squeeze(id))
        self.corners_cv = corners  # Save it to save on conversion. Is this worth it?
        self.process_corners()
    def process_corners(self):
        tmp = self.corners_cv.squeeze()
        self.center = Coordinate(*tmp.mean(axis=0))
        self.TL = Coordinate(tmp[0][0], tmp[0][1])
        self.TR = Coordinate(tmp[1][0], tmp[1][1])
        self.BR = Coordinate(tmp[2][0], tmp[2][1])
        self.BL = Coordinate(tmp[3][0], tmp[3][1])        

    @staticmethod
    def _test_marker_class():  # TEMPORARY - add testing package at some point
        # --- Simulated OpenCV output ---
        corners = (
            np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32),
            np.array([[[2, 2], [3, 2], [3, 3], [2, 3]]], dtype=np.float32),
        )
        ids = np.array([[9], [7]], dtype=np.int32)

        # --- Convert to Marker objects ---
        markers = Marker.from_cv_collection(ids, corners)

        assert len(markers) == 2

        # Marker 0
        m0 = markers[0]
        assert m0.id == 9
        assert (m0.TL.X, m0.TL.Y) == (0, 0)
        assert (m0.TR.X, m0.TR.Y) == (1, 0)
        assert (m0.BR.X, m0.BR.Y) == (1, 1)
        assert (m0.BL.X, m0.BL.Y) == (0, 1)

        # Marker 1
        m1 = markers[1]
        assert m1.id == 7
        assert (m1.TL.X, m1.TL.Y) == (2, 2)
        assert (m1.TR.X, m1.TR.Y) == (3, 2)
        assert (m1.BR.X, m1.BR.Y) == (3, 3)
        assert (m1.BL.X, m1.BL.Y) == (2, 3)

        H = np.identity(3)

        t_pts = cv.perspectiveTransform(corners[0], H)

        print(t_pts)

        # --- Convert back to OpenCV format ---
        corners_cv, ids_cv = Marker.to_cv_collection(markers)

        assert len(corners_cv) == 2
        assert ids_cv.shape == (2, 1)

        assert corners_cv[0].shape == (1, 4, 2)
        assert corners_cv[1].shape == (1, 4, 2)

        assert ids_cv[0, 0] == 9
        assert ids_cv[1, 0] == 7

        # --- Final OpenCV compatibility check ---
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = cv.aruco.drawDetectedMarkers(frame, corners_cv, ids_cv)

        assert frame is not None


class TrackedMarker:
    def __init__(self, marker: Marker):
        self.marker = marker
        self.missed_frames = 0
        # Further relevant states e.g. last_position for interpolation

    def update_seen(self, marker: Marker):
        self.marker = marker
        self.missed_frames = 0

    def update_unseen(self):
        self.missed_frames += 1


class MarkerTracker:
    def __init__(self, grace_frames=0):
        self.tracked_markers: dict[int, TrackedMarker] = {}
        self.grace_frames = grace_frames
    
    def update(self, detected_markers: list[Marker]):
            detected_map = {m.id: m for m in detected_markers}

            # Update existing tracked markers
            for tid, track in self.tracked_markers.items():
                if tid in detected_map:
                    track.update_seen(detected_map[tid])
                else:
                    track.update_unseen()

            # Add new tracked markers
            for mid, marker in detected_map.items():
                if mid not in self.tracked_markers:
                    self.tracked_markers[mid] = TrackedMarker(marker)

            # Remove stale tracked markers
            for tid in list(self.tracked_markers.keys()):
                if self.tracked_markers[tid].missed_frames > self.grace_frames:
                    del self.tracked_markers[tid]

    def get_tracked_markers(self):
        return [t.marker for t in self.tracked_markers.values()]


class ArucoMarkerDetector:
    """Class for detecting ArUco markers in images using specific parameters."""
    def __init__(self, aruco_dict: str, detector_params: dict = None) -> None:
        self.aruco_dict = cv.aruco.getPredefinedDictionary(getattr(cv.aruco, aruco_dict))
        
        # setup marker detector
        if detector_params is None:
            self.detector = cv.aruco.ArucoDetector(self.aruco_dict)
        else:
            self.detector_params = cv.aruco.DetectorParameters()
            for key, value in detector_params.items():
                setattr(self.detector_params, key, value)
            self.detector = cv.aruco.ArucoDetector(self.aruco_dict, self.detector_params)

    def detect(self, img: np.ndarray, debug: bool = False) -> tuple:
        """Detects ArUco markers in the given image and returns their corners and IDs."""
        # corners shape: N-sized tuple of (1, 4, 2) entries
        # ids shape: (N, 1)
        corners, ids, _ = self.detector.detectMarkers(img)
        if debug:
            debug_img = img.copy()
            if ids is not None:
                debug_img = cv.aruco.drawDetectedMarkers(debug_img, corners, ids)
            cv.imshow("Detected ArUco Markers", cv.resize(debug_img, (0, 0), fx=0.4, fy=0.4)) # TODO: adjust this to actual screen size
            print("Displaying debug window. To continue, select the debug window and then press any key.")
            cv.waitKey(0)
            cv.destroyWindow("Detected ArUco Markers")        
        return corners, ids