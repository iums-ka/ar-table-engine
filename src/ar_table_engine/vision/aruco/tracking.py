from enum import Enum, auto

from .marker import Marker


class TrackingStatus(Enum):
    UNSTABLE = auto()
    STABLE = auto()
    STALE = auto()


class TrackedMarker:
    def __init__(self, marker: Marker):
        self.tracking_status = TrackingStatus.UNSTABLE
        self.observed_frames = 0
        self.unobserved_frames = 0

        self.marker = marker


class MarkerTracker:
    def __init__(self, grace_frames=0, min_observations=1):
        self._tracked_markers: dict[int, TrackedMarker] = {}
        self._grace_frames = grace_frames
        self._min_observations = min_observations

    def update(self, detected_markers: list[Marker]):
        detected_map = {m.id: m for m in detected_markers}

        # Update currently tracked markers
        for tid, tm in self._tracked_markers.items():
            if tid in detected_map:
                self._update_observed(tm, detected_map[tid])
            else:
                self._update_unobserved(tm)
            self._update_state(tm)

        # Add new tracked markers
        for mid, m in detected_map.items():
            if mid not in self._tracked_markers:
                tm = TrackedMarker(m)
                self._tracked_markers[mid] = tm
                self._update_observed(tm, m)
                self._update_state(tm)

        self._cull()

    def _update_observed(self, tm: TrackedMarker, detected_marker: Marker):
        tm.marker = detected_marker
        tm.unobserved_frames = 0
        tm.observed_frames = min(tm.observed_frames + 1, self._min_observations)

    def _update_unobserved(self, tm: TrackedMarker):
        tm.observed_frames = 0
        tm.unobserved_frames += 1  # No need to cap since stale markers are culled

    def _update_state(self, tm: TrackedMarker):
        match tm.tracking_status:
            case TrackingStatus.UNSTABLE:
                if tm.observed_frames >= self._min_observations:
                    tm.tracking_status = TrackingStatus.STABLE
            case TrackingStatus.STABLE:
                if tm.unobserved_frames > self._grace_frames:
                    tm.tracking_status = TrackingStatus.STALE

    def _cull(self):
        stale_ids = [
            tid
            for tid, tm in self._tracked_markers.items()
            if tm.tracking_status == TrackingStatus.STALE
        ]

        for tid in stale_ids:
            del self._tracked_markers[tid]

    def get_stable_markers(self):
        return [
            tm.marker
            for tm in self._tracked_markers.values()
            if tm.tracking_status == TrackingStatus.STABLE
        ]

    def get_tracked_markers(self):
        return [tm for tm in self._tracked_markers.values()]
