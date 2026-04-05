# detector/speed_estimator.py
import time
import numpy as np
from collections import defaultdict, deque


class SpeedEstimator:
    """
    Estimates vehicle speed in km/h from centroid pixel displacement.
    pixels_per_meter: calibration — how many px = 1 real metre at mid-frame.
    """
    def __init__(self, pixels_per_meter=40.0):
        self.ppm = pixels_per_meter
        self._history = defaultdict(lambda: deque(maxlen=15))  # id → deque of (t, cx, cy)
        self._speeds = {}

    def update(self, oid, cx, cy):
        self._history[oid].append((time.monotonic(), cx, cy))
        h = self._history[oid]
        if len(h) < 3:
            self._speeds[oid] = 0.0
            return
        t0, x0, y0 = h[0]
        t1, x1, y1 = h[-1]
        dt = t1 - t0
        if dt < 1e-6:
            return
        px_dist = np.hypot(x1 - x0, y1 - y0)
        speed_kmh = (px_dist / self.ppm / dt) * 3.6
        self._speeds[oid] = min(round(speed_kmh, 1), 200.0)

    def get(self, oid):
        return self._speeds.get(oid, 0.0)

    def cleanup(self, active_ids):
        stale = set(self._history) - set(active_ids)
        for oid in stale:
            self._history.pop(oid, None)
            self._speeds.pop(oid, None)

    def summary_stats(self):
        speeds = [s for s in self._speeds.values() if s > 1.0]
        if not speeds:
            return {"avg_speed": 0.0, "max_speed": 0.0, "speeding_count": 0}
        return {
            "avg_speed": round(np.mean(speeds), 1),
            "max_speed": round(max(speeds), 1),
            "speeding_count": sum(1 for s in speeds if s > 30),
        }