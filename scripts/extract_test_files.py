import os
import shutil
import math

def move_subset_uniform(original_root, mask_root, output_original_root, output_mask_root, fraction=0.2):
    for dirpath, _, filenames in os.walk(original_root):
        if not filenames:
            continue

        # Sort to ensure consistency
        filenames.sort()

        # Preserve relative path
        rel_path = os.path.relpath(dirpath, original_root)
        mask_dir = os.path.join(mask_root, rel_path)

        # Output dirs
        out_orig_dir = os.path.join(output_original_root, rel_path)
        out_mask_dir = os.path.join(output_mask_root, rel_path)
        os.makedirs(out_orig_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # Number of files to move
        n_select = max(1, math.floor(len(filenames) * fraction))
        
        # Uniformly pick files (every k-th)
        step = len(filenames) / n_select
        selected_files = [filenames[round(i * step)] for i in range(n_select)]

        for file in selected_files:
            orig_file = os.path.join(dirpath, file)
            mask_file = os.path.join(mask_dir, file)

            if os.path.exists(orig_file) and os.path.exists(mask_file):
                shutil.move(orig_file, os.path.join(out_orig_dir, file))
                shutil.move(mask_file, os.path.join(out_mask_dir, file))
            else:
                print(f"⚠️ Missing mask or original for: {file}")

# Example usage
original_images_path = "D:\\ml_projects\\tinycrack\\data\\images\\Walls"
mask_images_path = "D:\\ml_projects\\tinycrack\\data\\masks\\Walls"
output_original_path = "D:\\ml_projects\\tinycrack\\data\\test_images\\Walls"
output_mask_path = "D:\\ml_projects\\tinycrack\\data\\test_masks\\Walls"

move_subset_uniform(original_images_path, mask_images_path, output_original_path, output_mask_path, fraction=0.2)
