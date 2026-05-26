import numpy as np

from ar_table_engine.vision.aruco.detection import Marker
from ar_table_engine.vision.aruco.tracking import MarkerTracker, TrackedMarkerState


# ----------------------------
# Helper: fake marker creator
# ----------------------------
def make_marker(mid: int) -> Marker:
    corners = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], dtype=np.float32)
    return Marker(mid, corners)


# ----------------------------
# Test 1: survives grace frames
# ----------------------------
def test_marker_survives_grace_period():
    tracker = MarkerTracker(grace_frames=3)

    m = make_marker(1)

    tracker.update([m])  # seen → STABLE (min_observations=1)
    tracker.update([])  # miss 1
    tracker.update([])  # miss 2
    tracker.update([])  # miss 3

    # still alive because condition is ">"
    assert 1 in tracker._tracked_markers
    assert tracker._tracked_markers[1].state == TrackedMarkerState.STABLE


# ----------------------------
# Test 2: removed after grace exceeded
# ----------------------------
def test_marker_removed_after_grace_exceeded():
    tracker = MarkerTracker(grace_frames=2)

    m = make_marker(1)

    tracker.update([m])  # seen
    tracker.update([])  # miss 1
    tracker.update([])  # miss 2
    tracker.update([])  # miss 3 → exceeds grace → STALE → culled

    assert 1 not in tracker._tracked_markers


# ----------------------------
# Test 3: re-seeing resets counter
# ----------------------------
def test_marker_reappears_resets_missed_frames():
    tracker = MarkerTracker(grace_frames=3)

    m = make_marker(1)

    tracker.update([m])  # seen
    tracker.update([])  # miss 1
    tracker.update([])  # miss 2

    tracker.update([m])  # seen again → reset

    assert tracker._tracked_markers[1].unobserved_frames == 0
    assert tracker._tracked_markers[1].state == TrackedMarkerState.STABLE


# ----------------------------
# Test 4: requires multiple observations to stabilize
# ----------------------------
def test_marker_requires_min_observations():
    tracker = MarkerTracker(min_observations=3)

    m = make_marker(1)

    tracker.update([m])  # 1 → INSTABLE
    assert tracker._tracked_markers[1].state == TrackedMarkerState.INSTABLE

    tracker.update([m])  # 2 → still INSTABLE
    assert tracker._tracked_markers[1].state == TrackedMarkerState.INSTABLE

    tracker.update([m])  # 3 → becomes STABLE
    assert tracker._tracked_markers[1].state == TrackedMarkerState.STABLE


# ----------------------------
# Test 5: instability resets if missed
# ----------------------------
def test_instability_resets_on_miss():
    tracker = MarkerTracker(min_observations=3)

    m = make_marker(1)

    tracker.update([m])  # 1
    tracker.update([m])  # 2

    tracker.update([])  # miss → observed resets

    tracker.update([m])  # 1 again

    assert tracker._tracked_markers[1].observed_frames == 1
    assert tracker._tracked_markers[1].state == TrackedMarkerState.INSTABLE
