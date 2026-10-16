import numpy as np
import pytest

from ar_table_engine.vision.aruco.marker import (
    Marker,
    markers_from_arrays,
    markers_to_arrays,
)
from ar_table_engine.vision.aruco.tracking import MarkerTracker, TrackingStatus


# ----------------------------
# Helper: fake marker creator
# ----------------------------
def make_marker(mid: int, offset=(0, 0), scale=1.0) -> Marker:
    x, y = offset
    corners = np.array(
        [
            [x, y],
            [x + scale, y],
            [x + scale, y + scale],
            [x, y + scale],
        ],
        dtype=np.float32,
    )
    return Marker(mid, corners)


# ----------------------------
# Test 1: survives grace frames
# ----------------------------
def test_marker_survives_grace_period():
    tracker = MarkerTracker(grace_frames=3)
    marker = make_marker(1)

    tracker.update([marker])  # Seen -> STABLE (min_observations=1)
    tracker.update([])  # Miss 1
    tracker.update([])  # Miss 2
    tracker.update([])  # Miss 3

    assert 1 in tracker._tracked_markers
    assert tracker._tracked_markers[1].tracking_status == TrackingStatus.STABLE


# ----------------------------
# Test 2: removed after grace exceeded
# ----------------------------
def test_marker_removed_after_grace_exceeded():
    tracker = MarkerTracker(grace_frames=2)
    marker = make_marker(1)

    tracker.update([marker])
    tracker.update([])  # Miss 1
    tracker.update([])  # Miss 2
    tracker.update([])  # Miss 3 -> STALE -> culled

    assert 1 not in tracker._tracked_markers


# ----------------------------
# Test 3: re-seeing resets counter
# ----------------------------
def test_marker_reappears_resets_missed_frames():
    tracker = MarkerTracker(grace_frames=3)
    marker = make_marker(1)

    tracker.update([marker])
    tracker.update([])
    tracker.update([])

    tracker.update([marker])

    tracked = tracker._tracked_markers[1]
    assert tracked.unobserved_frames == 0
    assert tracked.tracking_status == TrackingStatus.STABLE


# ----------------------------
# Test 4: requires multiple observations to stabilize
# ----------------------------
def test_marker_requires_min_observations():
    tracker = MarkerTracker(min_observations=3)
    marker = make_marker(1)

    tracker.update([marker])  # Observation 1
    tracked = tracker._tracked_markers[1]
    assert tracked.tracking_status == TrackingStatus.UNSTABLE

    tracker.update([marker])  # Observation 2
    assert tracked.tracking_status == TrackingStatus.UNSTABLE

    tracker.update([marker])  # Observation 3
    assert tracked.tracking_status == TrackingStatus.STABLE


# ----------------------------
# Test 5: instability resets if missed
# ----------------------------
def test_instability_resets_on_miss():
    tracker = MarkerTracker(min_observations=3)
    marker = make_marker(1)

    tracker.update([marker])  # Observation 1
    tracker.update([marker])  # Observation 2

    tracker.update([])  # Miss -> observation count resets
    tracker.update([marker])  # Observation 1 again

    tracked = tracker._tracked_markers[1]
    assert tracked.observed_frames == 1
    assert tracked.tracking_status == TrackingStatus.UNSTABLE


# ----------------------------
# Test 6: only stable markers are returned
# ----------------------------
def test_get_tracked_markers_returns_only_stable_markers():
    tracker = MarkerTracker(min_observations=2)

    tracker.update([make_marker(1), make_marker(2)])

    # Neither marker has enough observations yet.
    assert tracker.get_stable_markers() == []

    tracker.update([make_marker(1)])

    tracked = tracker.get_stable_markers()
    assert len(tracked) == 1
    assert tracked[0].state.id == 1
    assert tracked[0].tracking_status == TrackingStatus.STABLE


# ----------------------------
# Test 7: orientation is normalized
# ----------------------------
def test_marker_orientation_is_unit_length():
    marker = make_marker(1, scale=10.0)

    orientation = marker.orientation

    assert orientation.shape == (2,)
    assert np.linalg.norm(orientation) == pytest.approx(1.0)
    np.testing.assert_allclose(orientation, [0.0, -1.0])


# ----------------------------
# Test 8: orientation is independent of marker scale
# ----------------------------
def test_marker_orientation_is_independent_of_scale():
    small_marker = make_marker(1, scale=10.0)
    large_marker = make_marker(2, scale=100.0)

    np.testing.assert_allclose(
        small_marker.orientation,
        large_marker.orientation,
    )


# ----------------------------
# Test 9: degenerate edge returns zero vector
# ----------------------------
def test_marker_orientation_handles_zero_length_edge():
    corners = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 0],
        ],
        dtype=np.float32,
    )
    marker = Marker(1, corners)

    np.testing.assert_array_equal(
        marker.orientation,
        np.zeros(2, dtype=np.float32),
    )


# ----------------------------
# Test 10: marker array conversions
# ----------------------------
def test_marker_array_round_trip():
    original = [
        make_marker(9, offset=(10, 20), scale=5.0),
        make_marker(7, offset=(30, 40), scale=8.0),
    ]

    ids, corners = markers_to_arrays(original)

    assert ids.shape == (2,)
    assert ids.dtype == np.int32
    assert corners.shape == (2, 4, 2)
    assert corners.dtype == np.float32

    restored = markers_from_arrays(corners, ids)

    assert [m.id for m in restored] == [9, 7]

    for before, after in zip(original, restored):
        np.testing.assert_allclose(before.corners, after.corners)
