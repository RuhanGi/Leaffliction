#!/usr/bin/env python3

import cv2
import os
from pathlib import Path
from transformations import (
    gaussian_blur,
    create_mask,
    roi_objects,
    analyze_object,
    pseudolandmarks,
    color_histogram
)
from display import display_transformations


class ImageTransformer:
    """Handle image transformations for leaf disease analysis"""

    def __init__(self, use_mask=False):
        self.use_mask = use_mask

    def transform_image(self, image_path):
        """Apply all transformations to an image"""
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Create mask once if needed
        mask = create_mask(image) if self.use_mask else None
        
        transformations = {
            'gaussian_blur': gaussian_blur(image),
            'mask': create_mask(image),
            'roi_objects': roi_objects(image, mask),
            'analyze_object': analyze_object(image, mask),
            'pseudolandmarks': pseudolandmarks(image, mask),
            'color_histogram': color_histogram(image, mask),
        }
        
        return image, transformations

    @staticmethod
    def display_transformations(image, transformations):
        """Display all transformations"""
        display_transformations(image, transformations)

    def save_transformations(self, image_path, output_dir):
        """Save all transformations to directory"""
        image, transformations = self.transform_image(image_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        file_ext = os.path.splitext(filename)[1]
        
        # Save original
        original_path = os.path.join(output_dir, f"{base_name}_Original{file_ext}")
        cv2.imwrite(original_path, image)
        
        # Save transformations
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_GaussianBlur{file_ext}"),
                   transformations['gaussian_blur'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_Mask{file_ext}"),
                   transformations['mask'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_RoiObjects{file_ext}"),
                   transformations['roi_objects'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_AnalyzeObject{file_ext}"),
                   transformations['analyze_object'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_Pseudolandmarks{file_ext}"),
                   transformations['pseudolandmarks'])
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_ColorHistogram{file_ext}"),
                   transformations['color_histogram'])
        
        print(f"✓ Saved 7 images for: {base_name}")

    def batch_transform(self, src_dir, dst_dir):
        """Process all images in a directory"""
        src_path = Path(src_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        
        images = [f for f in src_path.iterdir() 
                 if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not images:
            print(f"No images found in {src_dir}")
            return
        
        print(f"Found {len(images)} images to process")
        success_count = 0
        error_count = 0
        
        for idx, image_path in enumerate(images, 1):
            try:
                print(f"[{idx}/{len(images)}] Processing: {image_path.name}")
                self.save_transformations(str(image_path), dst_dir)
                success_count += 1
            except Exception as e:
                print(f"✗ Error processing {image_path.name}: {e}")
                error_count += 1
        
        print(f"\n{'='*50}")
        print(f"Processing complete!")
        print(f"Success: {success_count} images")
        print(f"Errors: {error_count} images")
        print(f"Output directory: {dst_dir}")
        print(f"{'='*50}")
