# detector/distance.py

def estimate_distance(bbox_height, known_height=1.7, focal_length=600):
    """
    Estimate distance of object from camera using bounding box height.

    bbox_height: height of bounding box in pixels
    known_height: average human height (meters)
    focal_length: camera focal length (approx)
    """

    if bbox_height <= 0:
        return None

    distance = (known_height * focal_length) / bbox_height
    return round(distance, 2)
