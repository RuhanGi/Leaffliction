#!/usr/bin/env python3

import cv2
import numpy as np


def create_mask(image):
    """Create a binary mask using color-based segmentation
    
    Args:
        image: Input BGR image
        
    Returns:
        Binary mask highlighting leaf regions
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green colors (leaves are green)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
