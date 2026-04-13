import numpy as np
from service.vision.aruco import Marker, MarkerTracker


# ----------------------------
# Helper: fake marker creator
# ----------------------------
def make_marker(mid: int) -> Marker:
    corners = np.array(
        [[[0, 0], [1, 0], [1, 1], [0, 1]]],
        dtype=np.float32
    )
    return Marker(mid, corners)


# ----------------------------
# Test 1: survives grace frames
# ----------------------------
def test_marker_survives_grace_period():
    tracker = MarkerTracker(grace_frames=3)

    m = make_marker(1)

    tracker.update([m])  # seen
    tracker.update([])   # miss 1
    tracker.update([])   # miss 2
    tracker.update([])   # miss 3

    # still alive (exactly at threshold behavior depends on your > or >= logic)
    assert 1 in tracker.tracked_markers


# ----------------------------
# Test 2: removed after grace exceeded
# ----------------------------
def test_marker_removed_after_grace_exceeded():
    tracker = MarkerTracker(grace_frames=2)

    m = make_marker(1)

    tracker.update([m])  # seen
    tracker.update([])   # miss 1
    tracker.update([])   # miss 2
    tracker.update([])   # miss 3 → should be removed

    assert 1 not in tracker.tracked_markers


# ----------------------------
# Test 3: re-seeing resets counter
# ----------------------------
def test_marker_reappears_resets_missed_frames():
    tracker = MarkerTracker(grace_frames=3)

    m = make_marker(1)

    tracker.update([m])   # seen
    tracker.update([])    # miss 1
    tracker.update([])    # miss 2

    tracker.update([m])   # seen again → reset

    assert tracker.tracked_markers[1].missed_frames == 0