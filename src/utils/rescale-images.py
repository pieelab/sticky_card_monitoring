"""
Rescale and Add Size Information to Crops

Resizes crops to 224x224 and adds size metadata to filenames.
Supports both individual folders and recursive directory scanning.

Usage:
    python rescale_crops.py -s source_dir -t target_dir
    python rescale_crops.py -s C:\crops -t C:\crops_rescaled --target_size 224 --recursive

Based on: https://github.com/darsa-group/size-aware-classification/blob/main/04_rescale-images.py
"""

import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import argparse


BLACK = [0, 0, 0]


def count_mask_pixels(image):
    """
    Count non-black pixels in image (excluding black background/padding).
    
    Parameters
    ----------
    image : np.ndarray
        Image array (BGR or RGB)
        
    Returns
    -------
    int
        Number of non-black pixels
    """
    # Convert to RGB if BGR
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    num_pixels = np.sum(np.all(image != [0, 0, 0], axis=-1))
    return num_pixels


def pad_scale(im, target_size):
    """
    Pad image to square then resize to target size.
    
    Parameters
    ----------
    im : np.ndarray
        Input image
    target_size : int
        Target size (e.g., 224)
        
    Returns
    -------
    tuple
        (resized_image, original_size, resized_mask_pixels)
    """
    h, w = im.shape[:2]
    
    # Pad to square
    if w > h:
        vertical_margin = (w - h) / 2
        top_margin = np.ceil(vertical_margin).astype(int)
        bottom_margin = np.floor(vertical_margin).astype(int)
        pad = cv2.copyMakeBorder(im, top_margin, bottom_margin, 0, 0, 
                                 cv2.BORDER_CONSTANT, value=BLACK)
    else:
        horizontal_margin = (h - w) / 2
        left_margin = np.ceil(horizontal_margin).astype(int)
        right_margin = np.floor(horizontal_margin).astype(int)
        pad = cv2.copyMakeBorder(im, 0, 0, left_margin, right_margin, 
                                 cv2.BORDER_CONSTANT, value=BLACK)
    
    # Resize
    resized_image = cv2.resize(pad, (target_size, target_size))
    resized_mask_pixels = count_mask_pixels(resized_image)
    
    return resized_image, im.shape[0:2], resized_mask_pixels


def find_crops_folders(root_dir):
    """
    Recursively find all 'crops' folders in directory tree.
    
    Parameters
    ----------
    root_dir : str
        Root directory to search
        
    Returns
    -------
    list
        List of (crops_folder_path, parent_folder_name) tuples
    """
    crops_folders = []
    
    for root, dirs, files in os.walk(root_dir):
        if 'crops' in dirs:
            crops_path = os.path.join(root, 'crops')
            # Get the parent folder name (scan name)
            parent_name = os.path.basename(root)
            crops_folders.append((crops_path, parent_name))
    
    return crops_folders


def process_crops(source_dir, target_dir, target_size=224, recursive=False):
    """
    Process crops by adding size information to filenames.
    
    Parameters
    ----------
    source_dir : str
        Source directory (containing crops folder or scan folders)
    target_dir : str
        Target directory for output
    target_size : int
        Target resize size (default 224)
    recursive : bool
        If True, recursively find all 'crops' folders
        If False, expect direct crops folder
    
    Returns
    -------
    int
        Total number of crops processed
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return 0
    
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Copy COCO metadata if it exists
    import shutil
    coco_source = source_path / 'coco_instances.json'
    if coco_source.exists():
        coco_target = target_path / 'coco_instances.json'
        shutil.copy2(coco_source, coco_target)
        print(f"✓ Copied COCO metadata: coco_instances.json")
    else:
        print(f"⚠ Warning: coco_instances.json not found in source directory")
    
    # Find crops folders
    if recursive:
        crops_folders = find_crops_folders(source_dir)
        if not crops_folders:
            print(f"WARNING: No 'crops' folders found in {source_dir}")
            return 0
        print(f"Found {len(crops_folders)} crops folders to process")
    else:
        # Direct crops folder
        crops_path = source_path / 'crops'
        if crops_path.exists():
            crops_folders = [(str(crops_path), 'crops')]
        else:
            print(f"ERROR: No 'crops' folder found in {source_dir}")
            return 0
    
    total_processed = 0
    
    # Process each crops folder
    for crops_source, parent_name in crops_folders:
        # Create target structure
        if recursive:
            crops_target = target_path / parent_name / 'crops'
        else:
            crops_target = target_path / 'crops'
        
        crops_target.mkdir(parents=True, exist_ok=True)
        
        # Copy metadata file for this scan if in recursive mode
        if recursive:
            scan_source_dir = Path(crops_source).parent
            # Look for metadata files
            metadata_files = list(scan_source_dir.glob('metadata_*.json'))
            for metadata_file in metadata_files:
                metadata_target = target_path / parent_name / metadata_file.name
                metadata_target.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(metadata_file, metadata_target)
        
        # Find all PNG files in this crops folder
        images = glob(os.path.join(crops_source, '*.png'))
        
        if not images:
            continue
        
        # Process each image
        for item_path in tqdm(images, desc=f"Processing {parent_name}", leave=False):
            try:
                item = os.path.basename(item_path)
                img = cv2.imread(item_path)
                
                if img is None:
                    print(f"  Error reading image {item_path}")
                    continue
                
                h, w, _ = img.shape
                
                # Rotate if necessary so longer side is horizontal
                if h > w:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    h, w = img.shape[:2]
                
                # Count pixels before processing
                mask_pixels = count_mask_pixels(img)
                
                # Pad and resize
                resized_image, original_size, resized_mask_pixels = pad_scale(img, target_size)
                
                # Create new filename with size info
                file_name, file_extension = os.path.splitext(item)
                new_file_name = f"{file_name}_{w}_{h}_{mask_pixels}_{resized_mask_pixels}{file_extension}"
                target_item_path = crops_target / new_file_name
                
                # Save
                cv2.imwrite(str(target_item_path), resized_image)
                total_processed += 1
                
            except Exception as e:
                print(f"  Error processing {item_path}: {e}")
                continue
    
    return total_processed


def main():
    parser = argparse.ArgumentParser(
        description='Rescale crops and add size information to filenames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single crops folder
  python rescale_crops.py -s crops -t crops_rescaled
  
  # Recursively process all scan folders
  python rescale_crops.py -s test_crops -t test_crops_rescaled --recursive
  
  # Custom target size
  python rescale_crops.py -s crops -t crops_rescaled --target_size 256 --recursive
        """
    )
    
    parser.add_argument('-s', '--source', type=str, required=True,
                        help='Source directory (containing crops folder or scan folders)')
    parser.add_argument('-t', '--target', type=str, required=True,
                        help='Target directory for rescaled crops')
    parser.add_argument('--target_size', type=int, default=224,
                        help='Target resize size (default: 224)')
    parser.add_argument('--recursive', action='store_true',
                        help='Recursively process all crops folders in subdirectories')
    
    args = parser.parse_args()
    
    total = process_crops(
        args.source, 
        args.target, 
        target_size=args.target_size,
        recursive=args.recursive
    )
    
    print(f" Processing complete")
    print(f"  Total crops processed: {total}")
    print(f"  Output directory: {args.target}")
    
    if total == 0:
        print("\n No crops were processed. Check:")
        print("  1. Source directory path is correct")
        print("  2. Directory contains 'crops' folders or subdirectories with 'crops'")
        print("  3. PNG files exist in the crops folders")


if __name__ == '__main__':
    main()