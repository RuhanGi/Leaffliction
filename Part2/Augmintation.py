import sys
import os
from PIL import Image
import math

def load_image(path):
    img = Image.open(path)
    if img is None:
        raise ValueError("Cannot read the image")
    return img

def save(img, base, suffix, ext, out_dir):
    name = f"{base}_{suffix}{ext}"
    img.save(os.path.join(out_dir, name))

def flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(img):
    return img.rotate(15, expand=False)

# x' = a*x + b*y + c
# y' = d*x + e*y + f
def skew(img):
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0.2, 0, 0, 1, 0)
    )

def shear(img):
    w, h = img.size
    return img.transform(
        (w, h), 
        Image.AFFINE, 
        (1, 0, 0, 0.3, 1, 0)
    )

def crop(img):
    """Crop center portion of image (80% of original)"""
    w, h = img.size
    crop_percent = 0.8
    new_w = int(w * crop_percent)
    new_h = int(h * crop_percent)
    
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    
    cropped = img.crop((left, top, right, bottom))
    # Resize back to original size
    return cropped.resize((w, h), Image.LANCZOS)

def main():
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print("Error: file not found")
        sys.exit(1)

    out_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    base, ext = os.path.splitext(filename)

    img = load_image(path)

    save(flip(img), base, "Flip", ext, out_dir)
    save(rotate(img), base, "Rotate", ext, out_dir)
    save(skew(img), base, "Skew", ext, out_dir)
    save(shear(img), base, "Shear", ext, out_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")