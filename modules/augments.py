import numpy as np
import cv2


def apply_skew(img):
    """
    Skew: Perspective tilt (simulates looking from the side).
    Fastest method: cv2.warpPerspective
    """
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    squeeze = int(cols * 0.2)
    pts2 = np.float32([[0, 0], [cols-squeeze, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (cols, rows))


def apply_shear(img, shear=0.2):
    """
    Shear: Slants the image horizontally.
    Fastest method: cv2.warpAffine
    """
    rows, cols = img.shape[:2]
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (int(cols*1.2), rows))


def apply_distortion(img):
    """
    Distortion: 'Glass' or 'Wavy' effect.
    Fastest method: np.meshgrid + cv2.remap
    """
    rows, cols = img.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_dist = x + (20 * np.sin(2 * np.pi * y / 150))
    y_dist = y + (20 * np.sin(2 * np.pi * x / 150))
    return cv2.remap(
        img, x_dist.astype(np.float32),
        y_dist.astype(np.float32),
        cv2.INTER_LINEAR
    )


def transform(imgs):
    data = {
        "Original":  imgs,
        "Flip":    [cv2.flip(img, 0) for img in imgs],
        "Rotate":
            [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in imgs],
        "Skew":      [apply_skew(img) for img in imgs],
        "Shear":     [apply_shear(img) for img in imgs],
        "Crop":      [img[50:-50, 50:-50] for img in imgs],
        "Distortion": [apply_distortion(img) for img in imgs]
    }
    return data
