"""
End-to-end pipeline: rescale crops -> train/val/test split -> hierarchical
stage1 / stage2 datasets, for training a Debris/Arthropod -> subclass
model pair.

Given a source directory with one flat folder per class:

    source_dir/
        Debris/
        Arthropod/          (generic / unidentified arthropod)
        SBW/
        SWD_male/
        SWD_parasitoid/

this script:

  1. Rescales every crop (pad to square + resize to --target-size,
     rotating so the longer side is horizontal first) and writes it into
     an intermediate directory, one flat folder per source class:

        rescaled_dir/
            Debris/
            Arthropod/
            SBW/
            SWD_male/
            SWD_parasitoid/

  2. Splits each class's rescaled files into train/val/test (shuffled,
     seeded). This split is done ONCE per source class, before any
     pooling, so the same physical crop always lands in the same split
     in both stage1 and stage2 -- the two stages stay evaluated on
     consistent data.

  3. Builds two output trees, in split/class layout:

        stage1_dest/                     (Debris vs. Arthropod)
            train/
                Debris/
                Arthropod/                <- pooled: Arthropod + SBW + SWD_male + SWD_parasitoid
            val/
                Debris/
                Arthropod/
            test/
                Debris/
                Arthropod/

        stage2_dest/                     (4-way: subclasses + unidentified)
            train/
                SBW/
                SWD_male/
                SWD_parasitoid/
                unidentified_arthropod/   <- pooled from the generic Arthropod class
            val/
                SBW/
                SWD_male/
                SWD_parasitoid/
                unidentified_arthropod/
            test/
                SBW/
                SWD_male/
                SWD_parasitoid/
                unidentified_arthropod/

Files are hard-linked from the rescaled intermediate directory into
stage1/stage2 where possible (falls back to copy across filesystems), so
disk usage isn't duplicated for every stage. Pass --copy to force plain
copies instead.

Usage
-----
python build_pipeline.py \
    --source-dir data \
    --stage1-dest stage1 \
    --stage2-dest stage2 \
    --subclasses SBW,SWD_male,SWD_parasitoid
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
BLACK = [0, 0, 0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source-dir", required=True,
                         help="Root dir containing one flat folder of crops per class")
    parser.add_argument("--rescaled-dir", default="rescaled_crops",
                         help="Intermediate dir to write rescaled crops into "
                              "(default: ./rescaled_crops)")
    parser.add_argument("--stage1-dest", required=True,
                         help="Output root for the Debris/Arthropod dataset")
    parser.add_argument("--stage2-dest", required=True,
                         help="Output root for the subclass dataset")

    parser.add_argument("--debris-class", default="Debris",
                         help="Folder name of the debris class in source-dir")
    parser.add_argument("--generic-arthropod-class", default="Arthropod",
                         help="Folder name of the generic/unidentified arthropod "
                              "class in source-dir. Pass '' if you don't have "
                              "one and stage1's Arthropod class should be made "
                              "up of the subclasses only.")
    parser.add_argument("--subclasses", required=True,
                         help="Comma-separated folder names of the specific "
                              "subclasses, e.g. 'SBW,SWD_male,SWD_parasitoid'")
    parser.add_argument("--stage2-unidentified-name", default="unidentified_arthropod",
                         help="Destination folder name in stage2-dest for the "
                              "pooled generic/unidentified arthropod class. "
                              "Pass '' to exclude it.")

    parser.add_argument("--target-size", type=int, default=224,
                         help="Square size to resize crops to (default: 224)")

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed for the train/val/test shuffle (default: 42)")

    parser.add_argument("--recursive", dest="recursive", action="store_true",
                         default=True,
                         help="Search subdirectories of each class folder for "
                              "images too (default: on)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")

    parser.add_argument("--add-size-info", dest="add_size_info", action="store_true",
                         default=True,
                         help="Append original/mask pixel size info to filenames "
                              "(default: on)")
    parser.add_argument("--no-size-info", dest="add_size_info", action="store_false")

    parser.add_argument("--copy", action="store_true",
                         help="Force plain file copies instead of hard links")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Rescaling (adapted from rescale_crops.py)
# ---------------------------------------------------------------------------

def count_mask_pixels(image):
    """Count non-black pixels in image (excluding black background/padding)."""
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return int(np.sum(np.all(image != [0, 0, 0], axis=-1)))


def pad_scale(im, target_size):
    """Pad image to square then resize to target size."""
    h, w = im.shape[:2]

    if w > h:
        vertical_margin = (w - h) / 2
        top = int(np.ceil(vertical_margin))
        bottom = int(np.floor(vertical_margin))
        pad = cv2.copyMakeBorder(im, top, bottom, 0, 0,
                                  cv2.BORDER_CONSTANT, value=BLACK)
    else:
        horizontal_margin = (h - w) / 2
        left = int(np.ceil(horizontal_margin))
        right = int(np.floor(horizontal_margin))
        pad = cv2.copyMakeBorder(im, 0, 0, left, right,
                                  cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(pad, (target_size, target_size))
    resized_mask_pixels = count_mask_pixels(resized_image)
    return resized_image, im.shape[0:2], resized_mask_pixels


def find_images(class_dir, recursive=True):
    """Find image files directly inside (or, if recursive, under) class_dir."""
    class_path = Path(class_dir)
    if recursive:
        files = [p for p in class_path.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in class_path.iterdir()
                 if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def rescale_class(class_name, src_files, dest_dir, target_size, add_size_info):
    """Rescale every file in src_files, writing flat into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for src_path in tqdm(src_files, desc=f"Rescaling {class_name}", leave=False):
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"  Warning: could not read {src_path}, skipping")
            continue

        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = img.shape[:2]

        mask_pixels = count_mask_pixels(img)
        resized_image, _orig_size, resized_mask_pixels = pad_scale(img, target_size)

        stem, ext = src_path.stem, src_path.suffix
        if add_size_info:
            new_name = f"{stem}_{w}_{h}_{mask_pixels}_{resized_mask_pixels}{ext}"
        else:
            new_name = f"{stem}{ext}"

        dest_path = dest_dir / new_name
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{counter}{ext}"
            counter += 1

        cv2.imwrite(str(dest_path), resized_image)
        written.append(dest_path)

    return written


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def split_files(files, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--train/val/test ratios must sum to 1.0, got {total}")

    files = list(files)
    random.Random(seed).shuffle(files)

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


# ---------------------------------------------------------------------------
# Linking/copying into stage1/stage2 trees (adapted from
# build_hierarchical_datasets.py)
# ---------------------------------------------------------------------------

def link_or_copy(src, dst, force_copy=False):
    if os.path.exists(dst):
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def populate_split_dir(files, dest_dir, force_copy):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in files:
        dst = dest_dir / Path(f).name
        link_or_copy(str(f), str(dst), force_copy=force_copy)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = cli_args()

    subclasses = [c.strip() for c in args.subclasses.split(",") if c.strip()]
    if not subclasses:
        raise ValueError("--subclasses must contain at least one class name")

    generic_arthropod_class = args.generic_arthropod_class.strip() or None
    stage2_unidentified_name = args.stage2_unidentified_name.strip() or None
    debris_class = args.debris_class

    all_classes = list(subclasses)
    if generic_arthropod_class:
        all_classes.append(generic_arthropod_class)
    all_classes.append(debris_class)

    for class_name in all_classes:
        src_dir = os.path.join(args.source_dir, class_name)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {src_dir}\n"
                f"--source-dir should point at the root containing one "
                f"folder per class."
            )

    # --- Step 1 + 2: rescale then split, per source class -----------------
    print("Step 1/2: rescaling crops and splitting train/val/test...\n")
    class_splits = {}
    for class_name in all_classes:
        src_dir = os.path.join(args.source_dir, class_name)
        images = find_images(src_dir, recursive=args.recursive)
        if not images:
            print(f"Warning: no images found for class '{class_name}' in {src_dir}")

        rescaled_dest = os.path.join(args.rescaled_dir, class_name)
        rescaled_files = rescale_class(
            class_name, images, rescaled_dest, args.target_size, args.add_size_info
        )

        splits = split_files(
            rescaled_files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        class_splits[class_name] = splits

        print(f"  {class_name}: {len(rescaled_files)} rescaled "
              f"-> train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])}")

    # --- Step 3: stage1 (Debris vs Arthropod) ------------------------------
    print("\nStep 3a: building stage1 (Debris vs. Arthropod)...")
    arthropod_source_classes = list(subclasses)
    if generic_arthropod_class:
        arthropod_source_classes.append(generic_arthropod_class)

    for split in SPLITS:
        debris_dest = os.path.join(args.stage1_dest, split, debris_class)
        n = populate_split_dir(class_splits[debris_class][split], debris_dest, args.copy)
        print(f"  {split}/{debris_class}: {n} files")

        pooled = []
        for c in arthropod_source_classes:
            pooled.extend(class_splits[c][split])
        arthropod_dest = os.path.join(args.stage1_dest, split, "Arthropod")
        n = populate_split_dir(pooled, arthropod_dest, args.copy)
        print(f"  {split}/Arthropod (pooled from {arthropod_source_classes}): {n} files")

    # --- Step 3: stage2 (subclasses + unidentified) ------------------------
    print("\nStep 3b: building stage2 (subclasses + unidentified)...")
    for subclass in subclasses:
        for split in SPLITS:
            dest = os.path.join(args.stage2_dest, split, subclass)
            n = populate_split_dir(class_splits[subclass][split], dest, args.copy)
            print(f"  {split}/{subclass}: {n} files")

    if generic_arthropod_class and stage2_unidentified_name:
        for split in SPLITS:
            dest = os.path.join(args.stage2_dest, split, stage2_unidentified_name)
            n = populate_split_dir(class_splits[generic_arthropod_class][split], dest, args.copy)
            print(f"  {split}/{stage2_unidentified_name} (from {generic_arthropod_class}): {n} files")

    print("\nDone.")
    print(f"Rescaled crops:   {args.rescaled_dir}")
    print(f"Stage 1 dataset:  {args.stage1_dest}")
    print(f"Stage 2 dataset:  {args.stage2_dest}")


if __name__ == "__main__":
    main()