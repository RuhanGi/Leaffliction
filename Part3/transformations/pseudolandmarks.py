#!/usr/bin/env python3

import cv2
from .mask import create_mask


def pseudolandmarks(image, mask=None):
    """Generate and visualize pseudolandmarks on the leaf
    
    Args:
        image: Input BGR image
        mask: Optional pre-computed mask
        
    Returns:
        Image with pseudolandmarks and convex hull drawn
    """
    if mask is None:
        mask = create_mask(image)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    landmark_image = image.copy()
    
    if not contours:
        return landmark_image
    
    # Get the largest contour (the leaf)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Draw pseudolandmarks at contour vertices
    for point in approx:
        x, y = point[0]
        cv2.circle(landmark_image, (x, y), 8, (0, 255, 255), -1)
    
    # Draw convex hull
    hull = cv2.convexHull(largest_contour)
    cv2.drawContours(landmark_image, [hull], 0, (255, 255, 0), 2)
    
    return landmark_image
