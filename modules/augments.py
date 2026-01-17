import numpy as np
import cv2


def apply_skew(img):
    """Skew: Perspective tilt."""
    rows, cols = img.shape[:2]
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    squeeze = int(cols * 0.2)
    pts2 = np.float32([[0, 0], [cols-squeeze, 0], [0, rows], [cols, rows]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (cols, rows))


def apply_shear(img):
    """Shear: Slants the image."""
    rows, cols = img.shape[:2]
    shear = 0.2
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (int(cols*1.2), rows))


def apply_distortion(img):
    """Distortion: Wavy effect."""
    rows, cols = img.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x_dist = x + (20 * np.sin(2 * np.pi * y / 150))
    y_dist = y + (20 * np.sin(2 * np.pi * x / 150))
    return cv2.remap(
        img,
        x_dist.astype(np.float32),
        y_dist.astype(np.float32),
        cv2.INTER_LINEAR
    )


def apply_crop(img):
    """Center Crop"""
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return img
    return img[50:-50, 50:-50]


def apply_flip(img):
    return cv2.flip(img, 0)


def apply_rotate(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def transform(imgs):
    """Original Dictionary logic for Grid Visualization"""
    data = {
        "Original":   imgs,
        "Flip":       [apply_flip(img) for img in imgs],
        "Rotate":     [apply_rotate(img) for img in imgs],
        "Skew":       [apply_skew(img) for img in imgs],
        "Shear":      [apply_shear(img) for img in imgs],
        "Crop":       [apply_crop(img) for img in imgs],
        "Distortion": [apply_distortion(img) for img in imgs]
    }
    return data


AvailableTransforms = [
    (apply_flip, "Flip"),
    (apply_rotate, "Rotate"),
    (apply_skew, "Skew"),
    (apply_shear, "Shear"),
    (apply_crop, "Crop"),
    (apply_distortion, "Distortion")
]
