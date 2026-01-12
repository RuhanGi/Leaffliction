#!/usr/bin/env python3

import cv2
from .mask import create_mask


def roi_objects(image, mask=None):
    """Extract and highlight regions of interest
    
    Args:
        image: Input BGR image
        mask: Optional pre-computed mask
        
    Returns:
        Image with contours drawn around detected regions
    """
    if mask is None:
        mask = create_mask(image)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for drawing
    roi_image = image.copy()
    cv2.drawContours(roi_image, contours, -1, (0, 255, 0), 2)
    
    return roi_image
