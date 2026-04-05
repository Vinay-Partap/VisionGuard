# utils/danger_zones.py
import json
import os
import cv2
import numpy as np

ZONES_FILE = "auth/danger_zones.json"


class DangerZone:
    def __init__(self, zone_id, name, points, color_bgr=(0, 0, 255), sensitivity="high"):
        self.zone_id = zone_id
        self.name = name
        self.points = points          # normalized (x,y) tuples 0.0–1.0
        self.color_bgr = color_bgr
        self.sensitivity = sensitivity
        self.active = True

    def to_dict(self):
        return {
            "zone_id": self.zone_id, "name": self.name,
            "points": self.points, "color_bgr": list(self.color_bgr),
            "sensitivity": self.sensitivity, "active": self.active,
        }

    @classmethod
    def from_dict(cls, d):
        z = cls(d["zone_id"], d["name"], [tuple(p) for p in d["points"]],
                tuple(d.get("color_bgr", [0, 0, 255])), d.get("sensitivity", "high"))
        z.active = d.get("active", True)
        return z


class DangerZoneManager:
    def __init__(self):
        self.zones = {}
        self._load()

    def _load(self):
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE) as f:
                for d in json.load(f):
                    z = DangerZone.from_dict(d)
                    self.zones[z.zone_id] = z

    def _save(self):
        os.makedirs(os.path.dirname(ZONES_FILE), exist_ok=True)
        with open(ZONES_FILE, "w") as f:
            json.dump([z.to_dict() for z in self.zones.values()], f, indent=2)

    def add_zone(self, name, points, color_bgr=(0, 0, 255), sensitivity="high"):
        zid = f"zone_{len(self.zones) + 1:03d}"
        self.zones[zid] = DangerZone(zid, name, points, color_bgr, sensitivity)
        self._save()
        return zid

    def remove_zone(self, zid):
        self.zones.pop(zid, None)
        self._save()

    def toggle_zone(self, zid):
        if zid in self.zones:
            self.zones[zid].active = not self.zones[zid].active
            self._save()

    def list_zones(self):
        return list(self.zones.values())

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _point_in_poly(self, nx, ny, poly):
        """Ray-casting point-in-polygon for normalized coords."""
        n, inside = len(poly), False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > ny) != (yj > ny)) and (nx < (xj - xi) * (ny - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def check(self, cx, cy, frame_w, frame_h):
        """Return list of active zones the pixel point (cx,cy) falls inside."""
        nx, ny = cx / frame_w, cy / frame_h
        return [z for z in self.zones.values()
                if z.active and len(z.points) >= 3
                and self._point_in_poly(nx, ny, z.points)]

    # ── Drawing ────────────────────────────────────────────────────────────────

    def draw_on_frame(self, frame):
        """Draw all active danger zones on a cv2 BGR frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for zone in self.zones.values():
            if not zone.active or len(zone.points) < 3:
                continue
            pts = np.array([(int(px * w), int(py * h)) for px, py in zone.points], np.int32)
            cv2.fillPoly(overlay, [pts], zone.color_bgr)
            cv2.polylines(frame, [pts], True, zone.color_bgr, 2)
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            label = f"[{zone.sensitivity.upper()}] {zone.name}"
            cv2.putText(frame, label, (cx - 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color_bgr, 2)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        return frame

    def default_zones(self):
        return [
            {"name": "Crosswalk", "points": [(0.2, 0.7), (0.8, 0.7), (0.8, 0.95), (0.2, 0.95)],
             "color_bgr": (0, 0, 255), "sensitivity": "high"},
            {"name": "School Zone", "points": [(0.0, 0.4), (0.35, 0.4), (0.35, 0.75), (0.0, 0.75)],
             "color_bgr": (0, 165, 255), "sensitivity": "medium"},
        ]