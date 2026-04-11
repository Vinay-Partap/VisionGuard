import numpy as np
import cv2


class HeatmapAccumulator:
    def __init__(self, decay=0.97, radius=50):
        self.decay = float(decay)
        self.radius = int(radius)
        self.grid = None
        self.height = 0
        self.width = 0
        self.total_points = 0

    def ensure_size(self, h, w):
        h, w = int(h), int(w)
        if h <= 0 or w <= 0:
            return

        if self.grid is None:
            self.grid = np.zeros((h, w), dtype=np.float32)
            self.height, self.width = h, w
            return

        if self.height == h and self.width == w:
            return

        # Preserve old heat when frame size changes
        new_grid = np.zeros((h, w), dtype=np.float32)
        copy_h = min(self.height, h)
        copy_w = min(self.width, w)
        if copy_h > 0 and copy_w > 0:
            new_grid[:copy_h, :copy_w] = self.grid[:copy_h, :copy_w]

        self.grid = new_grid
        self.height, self.width = h, w

    def update(self, centroids, heat_val=1.0):
        if centroids is None:
            centroids = []

        if self.grid is None:
            if centroids:
                max_cx = max(int(c[0]) for c in centroids)
                max_cy = max(int(c[1]) for c in centroids)
                self.ensure_size(max(max_cy + 100, 480), max(max_cx + 100, 640))
            else:
                return

        self.grid *= self.decay
        radius = self.radius
        sigma = max(radius / 2.5, 1.0)

        for cx, cy in centroids:
            cx, cy = int(cx), int(cy)

            x0 = max(cx - radius, 0)
            x1 = min(cx + radius + 1, self.width)
            y0 = max(cy - radius, 0)
            y1 = min(cy + radius + 1, self.height)

            if x0 >= x1 or y0 >= y1:
                continue

            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            xx, yy = np.meshgrid(xs, ys)

            gauss = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
            self.grid[y0:y1, x0:x1] += gauss * float(heat_val)
            self.total_points += 1

        mx = float(self.grid.max())
        if mx > 0.0:
            self.grid /= mx

    def _draw_hotspot_boxes(
        self,
        image,
        thresh=0.6,
        min_area=200,
        color=(0, 255, 255),
        thickness=2,
    ):
        if self.grid is None:
            return image

        mask = (self.grid >= float(thresh)).astype(np.uint8) * 255
        if mask.max() == 0:
            return image

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < float(min_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, int(thickness))

        return image

    def composite_on(
        self,
        frame,
        alpha=0.55,
        show_boxes=True,
        box_thresh=0.6,
        box_min_area=200,
        box_color=(0, 255, 255),
        box_thickness=2,
    ):
        """Blend heatmap onto cv2 BGR frame and optionally draw hotspot boxes."""
        h, w = frame.shape[:2]
        self.ensure_size(h, w)

        if self.grid is None or self.grid.max() < 0.001:
            return frame

        heat_u8 = (self.grid * 255.0).clip(0, 255).astype(np.uint8)
        coloured = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

        heat_alpha = (self.grid * float(alpha)).clip(0, 1)[:, :, None]
        blended = (
            coloured.astype(np.float32) * heat_alpha
            + frame.astype(np.float32) * (1.0 - heat_alpha)
        ).clip(0, 255).astype(np.uint8)

        if show_boxes:
            blended = self._draw_hotspot_boxes(
                blended,
                thresh=box_thresh,
                min_area=box_min_area,
                color=box_color,
                thickness=box_thickness,
            )

        return blended

    def reset(self):
        self.grid = None
        self.height = 0
        self.width = 0
        self.total_points = 0

    def stats(self):
        if self.grid is None:
            return {"max_heat": 0, "mean_heat": 0, "hotspot_pct": 0, "total": 0}

        return {
            "max_heat": round(float(self.grid.max()), 3),
            "mean_heat": round(float(self.grid.mean()), 5),
            "hotspot_pct": round(float((self.grid > 0.5).mean() * 100), 2),
            "total": int(self.total_points),
        }