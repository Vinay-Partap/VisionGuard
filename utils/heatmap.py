# utils/heatmap.py  (main branch — uses cv2/numpy, no PIL)
import numpy as np
import cv2


class HeatmapAccumulator:
    """
    Accumulates detection centroids and renders a colour heatmap overlay
    directly on a cv2 BGR frame.
    """
    def __init__(self, decay=0.97):
        self.decay = decay
        self.grid = None   # initialised on first frame
        self.total_points = 0

    def _init_grid(self, h, w):
        if self.grid is None or self.grid.shape != (h, w):
            self.grid = np.zeros((h, w), dtype=np.float32)

    def update(self, centroids, heat_val=1.0):
        """centroids: list of (cx, cy) pixel coords."""
        if self.grid is None:
            return
        self.grid *= self.decay
        radius = 40
        for cx, cy in centroids:
            cx, cy = int(cx), int(cy)
            x0, x1 = max(cx - radius, 0), min(cx + radius, self.grid.shape[1])
            y0, y1 = max(cy - radius, 0), min(cy + radius, self.grid.shape[0])
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            xx, yy = np.meshgrid(xs, ys)
            gauss = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * (radius / 3)**2))
            self.grid[y0:y1, x0:x1] += gauss * heat_val
            self.total_points += 1
        mx = self.grid.max()
        if mx > 0:
            self.grid /= max(mx, 5.0)

    def composite_on(self, frame, alpha=0.5):
        """
        Blend heatmap onto a cv2 BGR frame and return result.
        Call _init_grid first with the frame dimensions.
        """
        h, w = frame.shape[:2]
        self._init_grid(h, w)

        if self.grid.max() < 0.01:
            return frame

        heat_u8 = (self.grid * 255).clip(0, 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        mask = (heat_u8 > 5).astype(np.float32)[:, :, None]
        blended = (coloured * mask * alpha + frame * (1 - mask * alpha)).astype(np.uint8)
        return blended

    def ensure_size(self, h, w):
        self._init_grid(h, w)

    def reset(self):
        self.grid = None
        self.total_points = 0

    def stats(self):
        if self.grid is None:
            return {"max_heat": 0, "mean_heat": 0, "hotspot_pct": 0, "total": 0}
        return {
            "max_heat": round(float(self.grid.max()), 3),
            "mean_heat": round(float(self.grid.mean()), 5),
            "hotspot_pct": round(float((self.grid > 0.7).mean() * 100), 2),
            "total": self.total_points,
        }