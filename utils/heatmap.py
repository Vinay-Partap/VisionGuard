# utils/heatmap.py  (main branch — cv2/numpy, no PIL)
import numpy as np
import cv2


class HeatmapAccumulator:
    """
    Accumulates detection centroids and renders a colour heatmap overlay
    directly on a cv2 BGR frame.
    """
    def __init__(self, decay=0.97):
        self.decay        = decay
        self.grid         = None   # float32 (H, W), raw accumulated values
        self.height       = 0
        self.width        = 0
        self.total_points = 0

    def ensure_size(self, h, w):
        """Initialise or resize the accumulation grid."""
        if self.grid is None or self.height != h or self.width != w:
            self.grid   = np.zeros((h, w), dtype=np.float32)
            self.height = h
            self.width  = w

    def update(self, centroids, heat_val=1.0):
        """centroids: list of (cx, cy) pixel coords."""
        if self.grid is None:
            if centroids:
                max_cx = max(c[0] for c in centroids)
                max_cy = max(c[1] for c in centroids)
                self.ensure_size(max(max_cy + 100, 480), max(max_cx + 100, 640))
            else:
                return

        self.grid *= self.decay
        radius = 50   # larger splat for better visibility

        for cx, cy in centroids:
            cx, cy = int(cx), int(cy)
            x0 = max(cx - radius, 0);  x1 = min(cx + radius, self.width)
            y0 = max(cy - radius, 0);  y1 = min(cy + radius, self.height)

            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            xx, yy = np.meshgrid(xs, ys)
            gauss = np.exp(
                -((xx - cx)**2 + (yy - cy)**2) / (2 * (radius / 2.5)**2)
            )
            self.grid[y0:y1, x0:x1] += gauss * heat_val
            self.total_points += 1

        # KEY FIX: normalise to actual max so brightest spot always = 1.0
        # Old code used max(mx, 5.0) which capped colours at 0.2 = nearly invisible
        mx = self.grid.max()
        if mx > 0:
            self.grid = self.grid / mx

    def composite_on(self, frame, alpha=0.6):
        """
        Blend heatmap onto a cv2 BGR frame and return result.
        frame: numpy array (H, W, 3) BGR
        """
        h, w = frame.shape[:2]
        self.ensure_size(h, w)

        if self.grid is None or self.grid.max() < 0.001:
            return frame

        # grid is normalised 0-1 so * 255 gives full colour range
        heat_u8  = (self.grid * 255).clip(0, 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        # Low threshold so even low-activity areas show blue tones
        mask = (heat_u8 > 2).astype(np.float32)[:, :, None]

        blended = (
            coloured * mask * alpha + frame * (1.0 - mask * alpha)
        ).clip(0, 255).astype(np.uint8)

        return blended

    def reset(self):
        self.grid         = None
        self.total_points = 0

    def stats(self):
        if self.grid is None:
            return {"max_heat": 0, "mean_heat": 0, "hotspot_pct": 0, "total": 0}
        return {
            "max_heat":    round(float(self.grid.max()), 3),
            "mean_heat":   round(float(self.grid.mean()), 5),
            "hotspot_pct": round(float((self.grid > 0.5).mean() * 100), 2),
            "total":       self.total_points,
        }