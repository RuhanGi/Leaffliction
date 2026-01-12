#!/usr/bin/env python3

import os
import sys
import argparse
from transformer import ImageTransformer


def main():
    parser = argparse.ArgumentParser(
        description="Image Transformation for Leaf Disease Analysis"
    )
    parser.add_argument('image', nargs='?', help='Path to image or directory')
    parser.add_argument('-src', '--source', help='Source directory for batch processing')
    parser.add_argument('-dst', '--destination', help='Destination directory for output')
    parser.add_argument('-mask', '--mask', action='store_true', 
                       help='Apply mask-based processing')
    
    args = parser.parse_args()
    
    if not args.image and not args.source:
        parser.print_help()
        sys.exit(1)
    
    transformer = ImageTransformer(use_mask=args.mask)
    
    try:
        #  -src and -dst
        if args.source and args.destination:
            if not os.path.isdir(args.source):
                print(f"Error: Source directory not found: {args.source}")
                sys.exit(1)
            transformer.batch_transform(args.source, args.destination)
        
        # Single image processing
        elif args.image:
            if os.path.isfile(args.image):
                image, transformations = transformer.transform_image(args.image)
                transformer.display_transformations(image, transformations)
            elif os.path.isdir(args.image):
                # If directory provided without -dst, display error
                print("Error: For directory processing, use -src and -dst flags")
                print("Example: ./Transformation.py -src directory/ -dst output_dir/")
                sys.exit(1)
            else:
                print(f"Error: File or directory not found: {args.image}")
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled error: {e}")
        sys.exit(1)