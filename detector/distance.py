KNOWN_WIDTH = 0.5
FOCAL_LENGTH = 700

def estimate_distance(x1, x2):
    pixel_width = x2 - x1
    if pixel_width == 0:
        return 999
    return (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

def is_near(distance, threshold=4.0):
    return distance < threshold
