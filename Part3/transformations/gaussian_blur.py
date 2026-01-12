#!/usr/bin/env python3

import cv2


def gaussian_blur(image):
    """Apply Gaussian blur to the image for noise reduction"""
    return cv2.GaussianBlur(image, (21, 21), 0)
