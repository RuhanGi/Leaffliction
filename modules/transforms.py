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
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def apply_mask(img):
    """
    Fig IV.3: Mask
    Keeps the original leaf colors but blacks out the background.
    """
    binary_mask = get_plant_mask(img)
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
    return masked_img


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


def transform(imgs):
    data = {
        # "Original":       imgs,
        "Gaussian Blur":  [cv2.GaussianBlur(img, (15, 15), 0) for img in imgs],
        "Mask":           [apply_mask(img) for img in imgs],
        # "ROI Objects":    [apply_roi(img) for img in imgs],
        # "Analyze Object": [apply_analyze(img) for img in imgs],
        # "Pseudolandmarks": [apply_landmarks(img) for img in imgs]
    }
    return data
