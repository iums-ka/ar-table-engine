import numpy as np


class Marker:
    def __init__(self, id, corners: np.ndarray):
        self.id = int(np.squeeze(id))
        # Store corners as (4, 2), independent of OpenCV's leading dimension.
        corners = np.asarray(corners, dtype=np.float32)
        self.corners = corners.reshape(4, 2)

    @property
    def TL(self):
        return self.corners[0]

    @property
    def TR(self):
        return self.corners[1]

    @property
    def BR(self):
        return self.corners[2]

    @property
    def BL(self):
        return self.corners[3]

    @property
    def center(self):
        return self.corners.mean(axis=0)

    @property
    def to_cv(self):
        return self.corners.reshape(1, 4, 2), np.array([[self.id]], dtype=np.int32)

    @property
    def orientation(self):
        edge = self.TL - self.BL
        length = np.linalg.norm(edge)

        if length == 0:
            return np.zeros(2, dtype=np.float32)

        return edge / length


def markers_to_arrays(markers: list[Marker]):
    """Convert Marker objects to (ids, corners) arrays."""
    ids = np.array([m.id for m in markers], dtype=np.int32)
    corners = np.array([m.corners for m in markers], dtype=np.float32)

    return ids, corners


def markers_from_arrays(corners, ids):
    """Convert ids and corners arrays back to Marker objects."""
    return [Marker(id, pts) for id, pts in zip(np.asarray(ids).reshape(-1), corners)]
