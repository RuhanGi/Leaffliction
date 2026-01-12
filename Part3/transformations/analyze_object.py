#!/usr/bin/env python3

import cv2
from .mask import create_mask


def analyze_object(image, mask=None):
    """Analyze objects and display their properties
    
    Args:
        image: Input BGR image
        mask: Optional pre-computed mask
        
    Returns:
        Image with bounding boxes and centroids marked
    """
    if mask is None:
        mask = create_mask(image)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    analyze_image = image.copy()
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        if area < 100:  # Filter small contours
            continue
        
        perimeter = cv2.arcLength(contour, True)
        
        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(analyze_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Draw circle at centroid
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(analyze_image, (cx, cy), 5, (0, 0, 255), -1)
    
    return analyze_image
