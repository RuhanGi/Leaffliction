#!/usr/bin/env python3

import cv2
import numpy as np


def display_transformations(image, transformations):
    """Display all transformations in a window
    
    Args:
        image: Original image
        transformations: Dictionary of transformation results
    """
    # Calculate appropriate window size based on screen
    max_width = 1280
    max_height = 720
    
    # Resize images to fit in a 3x2 grid
    tile_w = max_width // 3
    tile_h = max_height // 2
    
    # Create output image
    output = np.ones((max_height, max_width, 3), dtype=np.uint8) * 255
    
    titles = ['Original', 'Gaussian Blur', 'Mask', 'ROI Objects', 'Analyze Object', 'Pseudolandmarks']
    images_list = [
        image,
        transformations['gaussian_blur'],
        cv2.cvtColor(transformations['mask'], cv2.COLOR_GRAY2BGR),
        transformations['roi_objects'],
        transformations['analyze_object'],
        transformations['pseudolandmarks']
    ]
    
    # Arrange in grid
    for idx, (title, img) in enumerate(zip(titles, images_list)):
        row = idx // 3
        col = idx % 3
        
        # Resize to fit tile
        resized = cv2.resize(img, (tile_w, tile_h))
        
        # Place in output
        y_start = row * tile_h
        x_start = col * tile_w
        output[y_start:y_start+tile_h, x_start:x_start+tile_w] = resized
        
        # Add title with better visibility
        cv2.rectangle(output, (x_start, y_start), (x_start + 200, y_start + 35), 
                     (255, 255, 255), -1)
        cv2.putText(output, title, (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Display histogram in a separate smaller window
    hist = transformations['color_histogram']
    hist_height = 300
    hist_width = int(hist.shape[1] * (hist_height / hist.shape[0]))
    hist_resized = cv2.resize(hist, (hist_width, hist_height))
    
    cv2.imshow('Image Transformations', output)
    cv2.imshow('Color Histogram', hist_resized)
    
    print("Press any key to close the windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
