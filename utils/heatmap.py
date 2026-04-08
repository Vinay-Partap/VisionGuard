# utils/heatmap.py  (deploy branch — PIL only, no cv2)
import numpy as np
from PIL import Image, ImageFilter


class HeatmapAccumulator:
    """
    Accumulates detection centroids and renders a colour heatmap overlay.
    PIL-only — works on Streamlit Cloud without cv2.
    """
    def __init__(self, decay=0.97):
        self.decay  = decay
        self.grid   = None   # float32 (H, W), initialised on first frame
        self.width  = 0
        self.height = 0
        self.total_points = 0

    def ensure_size(self, h, w):
        if self.grid is None or self.height != h or self.width != w:
            self.grid   = np.zeros((h, w), dtype=np.float32)
            self.height = h
            self.width  = w

    def update(self, centroids, heat_val=1.0):
        """centroids: list of (cx, cy) pixel coords."""
        if self.grid is None:
            return
        self.grid *= self.decay
        radius = 40
        for cx, cy in centroids:
            cx, cy = int(cx), int(cy)
            x0 = max(cx - radius, 0);  x1 = min(cx + radius, self.width)
            y0 = max(cy - radius, 0);  y1 = min(cy + radius, self.height)
            xs = np.arange(x0, x1);   ys = np.arange(y0, y1)
            xx, yy = np.meshgrid(xs, ys)
            gauss = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * (radius / 3)**2))
            self.grid[y0:y1, x0:x1] += gauss * heat_val
            self.total_points += 1
        mx = self.grid.max()
        if mx > 0:
            self.grid /= max(mx, 5.0)

    def composite_on(self, pil_frame, alpha=0.5):
        """
        Blend heatmap onto a PIL RGB image and return result.
        pil_frame: PIL Image (RGB)
        """
        w, h = pil_frame.size
        self.ensure_size(h, w)

        if self.grid.max() < 0.01:
            return pil_frame

        heat = self.grid.clip(0, 1)
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Colour bands: blue → cyan → yellow → red
        m1 = (heat >= 0.01) & (heat < 0.25)
        m2 = (heat >= 0.25) & (heat < 0.5)
        m3 = (heat >= 0.5)  & (heat < 0.75)
        m4 =  heat >= 0.75

        t1 = heat / 0.25
        rgba[m1, 2] = (200 * t1[m1]).astype(np.uint8)
        rgba[m1, 3] = (140 * t1[m1] * alpha).astype(np.uint8)

        t2 = (heat - 0.25) / 0.25
        rgba[m2, 1] = np.clip(100 + 155 * t2[m2], 0, 255).astype(np.uint8)
        rgba[m2, 2] = 200
        rgba[m2, 3] = int(alpha * 180)

        t3 = (heat - 0.5) / 0.25
        rgba[m3, 0] = (255 * t3[m3]).astype(np.uint8)
        rgba[m3, 1] = 255
        rgba[m3, 2] = (200 * (1 - t3[m3])).astype(np.uint8)
        rgba[m3, 3] = int(alpha * 210)

        rgba[m4, 0] = 255
        rgba[m4, 1] = (255 * (1 - (heat[m4] - 0.75) / 0.25)).astype(np.uint8)
        rgba[m4, 3] = int(alpha * 255)

        heatmap_img = Image.fromarray(rgba, mode="RGBA")
        heatmap_img = heatmap_img.filter(ImageFilter.GaussianBlur(radius=4))

        base = pil_frame.convert("RGBA")
        composite = Image.alpha_composite(base, heatmap_img)
        return composite.convert("RGB")

    def reset(self):
        self.grid   = None
        self.total_points = 0

    def stats(self):
        if self.grid is None:
            return {"max_heat": 0, "mean_heat": 0, "hotspot_pct": 0, "total": 0}
        return {
            "max_heat":    round(float(self.grid.max()), 3),
            "mean_heat":   round(float(self.grid.mean()), 5),
            "hotspot_pct": round(float((self.grid > 0.7).mean() * 100), 2),
            "total":       self.total_points,
        }