#!/usr/bin/env python3

import cv2
import numpy as np
from .mask import create_mask


def color_histogram(image, mask=None):
    """Create color histogram visualization
    
    Args:
        image: Input BGR image
        mask: Optional pre-computed mask
        
    Returns:
        Image showing color histograms for each channel
    """
    if mask is None:
        mask = create_mask(image)
    
    # Apply mask to get only leaf region
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Calculate histograms for each channel
    colors = ('b', 'g', 'r')
    hist_image = np.zeros((256, 256 * 3, 3), dtype=np.uint8)
    hist_image[:] = 255  # White background
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([masked_image], [i], mask, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten() * 255
        
        # Draw histogram
        start_x = i * 256
        for j in range(255):
            cv2.line(hist_image, (start_x + j, 255 - int(hist[j])),
                    (start_x + j + 1, 255 - int(hist[j + 1])),
                    (255 if color == 'b' else 0, 255 if color == 'g' else 0,
                     255 if color == 'r' else 0), 2)
    
    return hist_image
