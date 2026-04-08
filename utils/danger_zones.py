# utils/danger_zones.py  (deploy branch — PIL only, no cv2)
import json
import os
from PIL import Image, ImageDraw

ZONES_FILE = "auth/danger_zones.json"


class DangerZone:
    def __init__(self, zone_id, name, points, color_hex="#FF0000", sensitivity="high"):
        self.zone_id     = zone_id
        self.name        = name
        self.points      = points        # normalised (x,y) tuples 0.0–1.0
        self.color_hex   = color_hex
        self.sensitivity = sensitivity
        self.active      = True

    def to_dict(self):
        return {
            "zone_id":     self.zone_id,
            "name":        self.name,
            "points":      self.points,
            "color_hex":   self.color_hex,
            "sensitivity": self.sensitivity,
            "active":      self.active,
        }

    @classmethod
    def from_dict(cls, d):
        # Support both old color_bgr format and new color_hex format
        if "color_hex" in d:
            color_hex = d["color_hex"]
        elif "color_bgr" in d:
            b, g, r = d["color_bgr"]
            color_hex = "#{:02X}{:02X}{:02X}".format(r, g, b)
        else:
            color_hex = "#FF0000"
        z = cls(d["zone_id"], d["name"],
                [tuple(p) for p in d["points"]],
                color_hex, d.get("sensitivity", "high"))
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

    def add_zone(self, name, points, color_hex="#FF4444", sensitivity="high"):
        zid = f"zone_{len(self.zones) + 1:03d}"
        self.zones[zid] = DangerZone(zid, name, points, color_hex, sensitivity)
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

    # ── Geometry ───────────────────────────────────────────────────────────

    def _point_in_poly(self, nx, ny, poly):
        """Ray-casting point-in-polygon for normalised coords."""
        n, inside = len(poly), False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > ny) != (yj > ny)) and \
               (nx < (xj - xi) * (ny - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    def check(self, cx, cy, frame_w, frame_h):
        """Return list of active zones the pixel point (cx,cy) falls inside."""
        nx, ny = cx / frame_w, cy / frame_h
        return [z for z in self.zones.values()
                if z.active and len(z.points) >= 3
                and self._point_in_poly(nx, ny, z.points)]

    # ── Drawing (PIL) ──────────────────────────────────────────────────────

    def draw_on_frame(self, pil_frame):
        """Draw all active danger zones on a PIL RGB image."""
        w, h   = pil_frame.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw    = ImageDraw.Draw(overlay)

        for zone in self.zones.values():
            if not zone.active or len(zone.points) < 3:
                continue

            pts = [(int(px * w), int(py * h)) for px, py in zone.points]
            hex_c = zone.color_hex.lstrip("#")
            r = int(hex_c[0:2], 16)
            g = int(hex_c[2:4], 16)
            b = int(hex_c[4:6], 16)

            draw.polygon(pts, fill=(r, g, b, 50))
            draw.line(pts + [pts[0]], fill=(r, g, b, 220), width=2)

            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            sens_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(zone.sensitivity, "")
            label = f"{sens_icon} {zone.name}"
            draw.rectangle([cx - 55, cy - 11, cx + 55, cy + 11], fill=(0, 0, 0, 160))
            draw.text((cx - 52, cy - 10), label, fill=(r, g, b, 255))

        base      = pil_frame.convert("RGBA")
        composite = Image.alpha_composite(base, overlay)
        return composite.convert("RGB")

    def default_zones(self):
        return [
            {"name": "Crosswalk",    "points": [(0.2, 0.7), (0.8, 0.7), (0.8, 0.95), (0.2, 0.95)],
             "color_hex": "#FF4444", "sensitivity": "high"},
            {"name": "School Zone",  "points": [(0.0, 0.4), (0.35, 0.4), (0.35, 0.75), (0.0, 0.75)],
             "color_hex": "#FFA500", "sensitivity": "medium"},
        ]