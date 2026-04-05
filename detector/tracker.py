# detector/tracker.py
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id = 0
        self.objects = OrderedDict()      # id → centroid
        self.labels = OrderedDict()       # id → label
        self.disappeared = OrderedDict()  # id → frames missing
        self.trajectories = OrderedDict() # id → list of (cx, cy)
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid, label):
        self.objects[self.next_id] = centroid
        self.labels[self.next_id] = label
        self.disappeared[self.next_id] = 0
        self.trajectories[self.next_id] = [centroid.tolist()]
        self.next_id += 1

    def _deregister(self, oid):
        for d in (self.objects, self.labels, self.disappeared, self.trajectories):
            d.pop(oid, None)

    def update(self, rects, labels):
        """
        rects  : list of (x1,y1,x2,y2)
        labels : list of label strings, same order as rects
        Returns: OrderedDict { id: (cx, cy, label) }
        """
        if len(rects) == 0:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self._output()

        input_centroids = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2]
            for x1, y1, x2, y2 in rects
        ], dtype=int)

        if len(self.objects) == 0:
            for i, c in enumerate(input_centroids):
                self._register(c, labels[i])
            return self._output()

        oids = list(self.objects.keys())
        existing = np.array(list(self.objects.values()))
        D = dist.cdist(existing, input_centroids)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.max_distance:
                continue
            oid = oids[r]
            self.objects[oid] = input_centroids[c]
            self.labels[oid] = labels[c]
            self.disappeared[oid] = 0
            self.trajectories[oid].append(input_centroids[c].tolist())
            if len(self.trajectories[oid]) > 60:
                self.trajectories[oid].pop(0)
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(len(oids))) - used_rows:
            oid = oids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self._deregister(oid)

        for c in set(range(len(rects))) - used_cols:
            self._register(input_centroids[c], labels[c])

        return self._output()

    def _output(self):
        return OrderedDict(
            (oid, (int(cx), int(cy), self.labels[oid]))
            for oid, (cx, cy) in self.objects.items()
        )

    def get_trajectory(self, oid):
        return self.trajectories.get(oid, [])