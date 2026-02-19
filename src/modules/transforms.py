import numpy as np
import cv2


def get_plant_mask(img):
    """
    Helper: Creates a binary mask where white = leaf, black = background.
    Used by ROI, Analyze, and Landmarks.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def apply_mask(img):
    """
    Fig IV.3: Mask
    Keeps the original leaf colors but blacks out the background.
    """
    binary_mask = get_plant_mask(img)
    return cv2.bitwise_and(img, img, mask=binary_mask)


def apply_roi(img):
    """
    Fig IV.4: Region of Interest. Draws a box around the leaf.
    """
    out = img.copy()
    mask = get_plant_mask(img)
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return out


def apply_analyze(img):
    """
    Fig IV.5: Analyze Object. Traces the outline (contour) of the leaf.
    """
    out = img.copy()
    mask = get_plant_mask(img)
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [c], -1, (0, 255, 0), 3)
    return out


def apply_landmarks(img):
    """
    Fig IV.6: Pseudolandmarks. Plots points along the leaf structure.
    """
    out = img.copy()
    mask = get_plant_mask(img)
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if cnts:
        c = max(cnts, key=cv2.contourArea)
        for i, point in enumerate(c):
            if i % 20 == 0:
                x, y = point[0]
                cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
    return out


def transform(imgs, selection=None):
    """
    Applies transformations.
    If 'selection' is provided, only applies that specific transform.
    """
    ops = {
        "Gaussian Blur": lambda x: cv2.GaussianBlur(x, (15, 15), 0),
        "Mask":            apply_mask,
        "ROI Objects":     apply_roi,
        "Analyze Object":  apply_analyze,
        "Pseudolandmarks": apply_landmarks
    }
    data = {"Original": imgs}

    for key, func in ops.items():
        if selection and selection.lower() not in key.lower():
            continue
        data[key] = [func(img) for img in imgs]

    return data
