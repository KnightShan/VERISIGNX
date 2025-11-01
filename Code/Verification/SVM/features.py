import numpy as np
import cv2
from skimage.measure import regionprops, label


def Ratio(img):
    """Fraction of white pixels in a binary image (0/255)."""
    white_count = np.count_nonzero(img == 255)
    total = img.shape[0] * img.shape[1]
    return white_count / total if total > 0 else 0


def Centroid(img):
    """Normalized centroid of white pixels (row, col)."""
    num_whites = np.count_nonzero(img == 255)
    if num_whites == 0:
        return 0.0, 0.0

    coords = np.argwhere(img == 255)  # (row, col) for each white pixel
    centroid = coords.mean(axis=0) / np.array(img.shape)
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    """
    Returns eccentricity & solidity of the largest region.
    img should be binary (0/255) but function is defensive: returns (0.0,0.0)
    if input is empty or contains no foreground.
    """
    # defensive checks
    if img is None:
        return 0.0, 0.0

    arr = np.asarray(img)
    # empty array or malformed shape
    if arr.size == 0:
        return 0.0, 0.0

    # If color, convert to single channel by summing (safe for binary masks)
    if arr.ndim == 3:
        # use first channel (or sum) — we assume non-zero indicates foreground
        arr = arr[..., 0]

    # If there is no foreground at all, return safe defaults
    if np.count_nonzero(arr) == 0:
        return 0.0, 0.0

    # label and compute regionprops safely
    try:
        labeled = label(arr > 0)
    except Exception:
        return 0.0, 0.0

    if labeled.size == 0:
        return 0.0, 0.0

    regions = regionprops(labeled)
    if not regions:
        return 0.0, 0.0

    # choose largest region (more robust than taking first)
    r = max(regions, key=lambda reg: reg.area)
    # regionprops fields eccentricity/solidity may exist — otherwise default
    ecc = getattr(r, "eccentricity", 0.0)
    sol = getattr(r, "solidity", 0.0)
    return float(ecc), float(sol)

def SkewKurtosis(img):
    """
    Skewness and kurtosis of horizontal/vertical pixel projections.
    img should be binary (0/255).
    """
    h, w = img.shape
    x = np.arange(w)  # columns
    y = np.arange(h)  # rows

    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)

    total = np.sum(img)
    if total == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Centroids
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)

    # Standard deviations
    sx = np.sqrt(np.sum(((x - cx) ** 2) * xp) / total)
    sy = np.sqrt(np.sum(((y - cy) ** 2) * yp) / total)

    if sx == 0 or sy == 0:
        return (0.0, 0.0), (0.0, 0.0)

    # Skewness
    skewx = np.sum(((x - cx) ** 3) * xp) / (total * sx ** 3)
    skewy = np.sum(((y - cy) ** 3) * yp) / (total * sy ** 3)

    # Kurtosis (excess kurtosis: subtract 3)
    kurtx = np.sum(((x - cx) ** 4) * xp) / (total * sx ** 4) - 3
    kurty = np.sum(((y - cy) ** 4) * yp) / (total * sy ** 4) - 3

    return (skewx, skewy), (kurtx, kurty)


def get_contour_features(im, display=False):
    """
    Aspect ratio of bounding rectangle, bounding rectangle area,
    convex hull area, and contour area.
    Input: binary image (0/255).
    """
    nonzero = cv2.findNonZero(im)
    if nonzero is None:
        return 0.0, 0.0, 0.0, 0.0

    rect = cv2.minAreaRect(nonzero)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)  # safer than np.int8

    w = np.linalg.norm(box[0] - box[1])
    h = np.linalg.norm(box[1] - box[2])
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    bounding_rect_area = w * h

    if display:
        image1 = cv2.drawContours(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), [box], 0, (120, 120, 120), 2)
        cv2.imshow("Bounding Box", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
        cv2.waitKey(0)

    hull = cv2.convexHull(nonzero)
    hull_area = cv2.contourArea(hull)

    try:
        contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        # OpenCV 3.x returns (image, contours, hierarchy)
        _, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

    return aspect_ratio, bounding_rect_area, hull_area, contour_area
