import os
import random
import shutil
import cv2
from .augments import AvailableTransforms

def balance_directory(train_paths, train_dir, target_count):
    needed = target_count - len(train_paths)
    if needed <= 0:
        return

    for i in range(needed):
        rand_path = random.choice(train_paths)
        img = cv2.imread(rand_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        func, name = random.choice(AvailableTransforms)
        aug_img = func(img_rgb)
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        
        base_name = os.path.splitext(os.path.basename(rand_path))[0]
        ext = os.path.splitext(rand_path)[1]
        save_path = os.path.join(train_dir, f"{base_name}_aug_{i}_{name}{ext}")
        
        cv2.imwrite(save_path, aug_img_bgr)

def build_pipeline(src_root, dst_root, ratio, target_count):
    for root, dirs, files in os.walk(src_root):
        imgs = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not imgs:
            continue
        
        rel_path = os.path.relpath(root, src_root)
        train_dir = os.path.join(dst_root, 'train', rel_path)
        val_dir = os.path.join(dst_root, 'val', rel_path)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        random.shuffle(imgs)
        split_idx = int(len(imgs) * ratio)
        train_files = imgs[:split_idx]
        val_files = imgs[split_idx:]
        
        for f in val_files:
            shutil.copy2(os.path.join(root, f), os.path.join(val_dir, f))
            
        train_paths = []
        for f in train_files:
            dst_path = os.path.join(train_dir, f)
            shutil.copy2(os.path.join(root, f), dst_path)
            train_paths.append(dst_path)
        
        print(f"[{rel_path}] Split: {len(train_files)} Train, {len(val_files)} Val")
        
        if target_count:
            balance_directory(train_paths, train_dir, target_count)
            print(f"[{rel_path}] Augmented to {target_count} total training images.")