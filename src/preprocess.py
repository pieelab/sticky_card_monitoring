"""
End-to-end pipeline: rescale crops -> train/val/test split -> hierarchical
stage1 / stage2 datasets, for training a Debris/Arthropod -> subclass
model pair.

Given a source directory with one flat folder per class:

    source_dir/
        Debris/
        Arthropod/               (generic / unidentified arthropod)
        Small_black_weevil/
        SWD_male/
        SWD_parasitoid/

this script:

  1. Rescales every crop (pad to square + resize to --target-size,
     rotating so the longer side is horizontal first) and writes it into
     an intermediate directory, one flat folder per source class.

  2. Splits each class's rescaled files into train/val/test (shuffled,
     seeded). This split is done ONCE per source class, before any
     pooling, so the same physical crop always lands in the same split
     in both stage1 and stage2 -- the two stages stay evaluated on
     consistent data.

  3. Builds two output trees, in split/class layout, using the class
     names and order hardcoded in the CLASS CONFIGURATION section below:

        stage1_dest/
            train/
                0_Debris/
                1_Arthropod/               <- pooled: Arthropod + all subclasses
            val/    (same layout)
            test/   (same layout)

        stage2_dest/
            train/
                0_SWD_male/
                1_SWD_parasitoid/
                2_Small_black_weevil/
                3_Unidentified_Arthropod/  <- pooled from the generic Arthropod class
            val/    (same layout)
            test/   (same layout)

Files are hard-linked from the rescaled intermediate directory into
stage1/stage2 where possible (falls back to copy across filesystems), so
disk usage isn't duplicated for every stage. Pass --copy to force plain
copies instead.

To change which source folders map to which output class names, or their
order/numbering, edit the CLASS CONFIGURATION constants below -- these are
no longer CLI flags.

Usage
-----
python build_pipeline.py \
    --source-dir data \
    --stage1-dest stage1 \
    --stage2-dest stage2
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
# CLASS CONFIGURATION -- edit this to match your source folder names and
# desired stage1 / stage2 output folder names/order. Each entry is
# (source_folder_name_under_source_dir, output_folder_name).
# ---------------------------------------------------------------------------

# Debris class: (source_name, dest_name). dest_name is its folder name in stage1.
DEBRIS_CLASS = ("Debris", "0_Debris")

# Generic/unidentified arthropod class: source folder name only. This pools
# into STAGE1_ARTHROPOD_NAME in stage1, and becomes its own class named
# STAGE2_UNIDENTIFIED_NAME in stage2. Set to None if you don't have this
# source class (stage1's Arthropod would then be subclasses-only, and
# stage2 would have no unidentified class).
GENERIC_ARTHROPOD_CLASS = "Arthropod"

# Output folder name for the pooled Arthropod class in stage1.
STAGE1_ARTHROPOD_NAME = "1_Arthropod"

# Output folder name for the generic/unidentified arthropod class in stage2.
# Set to None to exclude it from stage2 entirely.
STAGE2_UNIDENTIFIED_NAME = "3_Unidentified_Arthropod"

# Specific arthropod subclasses: (source_name, dest_name) pairs, in the
# order you want them to appear in stage2.
SUBCLASSES = [
    ("SWD_male", "0_SWD_male"),
    ("SWD_parasitoid", "1_SWD_parasitoid"),
    ("Small_black_weevil", "2_Small_black_weevil"),
]


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

    parser.add_argument("--cleanup-rescaled", action="store_true",
                         help="Delete --rescaled-dir after stage1/stage2 are built. "
                              "Only frees real disk space if --copy was used (or "
                              "hard-linking fell back to copying, e.g. across "
                              "filesystems) -- otherwise the rescaled files and "
                              "the stage1/stage2 files share the same underlying "
                              "data via hard links, and this just tidies up paths.")

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

    debris_source, debris_dest = DEBRIS_CLASS
    generic_arthropod_class = GENERIC_ARTHROPOD_CLASS
    stage2_unidentified_name = STAGE2_UNIDENTIFIED_NAME
    subclass_pairs = SUBCLASSES

    if not subclass_pairs:
        raise ValueError("SUBCLASSES must contain at least one (source, dest) pair")

    # (source_name, dest_name) pairs for every class we need to rescale/split.
    # generic_arthropod_class is never renamed at this stage -- it only ever
    # surfaces in the output as pooled STAGE1_ARTHROPOD_NAME (stage1) or
    # STAGE2_UNIDENTIFIED_NAME (stage2), so source == dest for it here.
    all_class_pairs = list(subclass_pairs)
    if generic_arthropod_class:
        all_class_pairs.append((generic_arthropod_class, generic_arthropod_class))
    all_class_pairs.append((debris_source, debris_dest))

    for source_name, _dest_name in all_class_pairs:
        src_dir = os.path.join(args.source_dir, source_name)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"Expected class directory not found: {src_dir}\n"
                f"--source-dir should point at the root containing one "
                f"folder per class. Check the CLASS CONFIGURATION constants "
                f"at the top of this script match your source folder names."
            )

    # --- Step 1 + 2: rescale then split, per source class -----------------
    print("Step 1/2: rescaling crops and splitting train/val/test...\n")
    class_splits = {}  # keyed by DEST name
    for source_name, dest_name in all_class_pairs:
        src_dir = os.path.join(args.source_dir, source_name)
        images = find_images(src_dir, recursive=args.recursive)
        if not images:
            print(f"Warning: no images found for class '{source_name}' in {src_dir}")

        rescaled_dest = os.path.join(args.rescaled_dir, dest_name)
        rescaled_files = rescale_class(
            dest_name, images, rescaled_dest, args.target_size, args.add_size_info
        )

        splits = split_files(
            rescaled_files, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        class_splits[dest_name] = splits

        label = dest_name if dest_name == source_name else f"{source_name} -> {dest_name}"
        print(f"  {label}: {len(rescaled_files)} rescaled "
              f"-> train={len(splits['train'])} val={len(splits['val'])} "
              f"test={len(splits['test'])}")

    subclass_dest_names = [dest for _source, dest in subclass_pairs]

    # --- Step 3: stage1 (Debris vs Arthropod) ------------------------------
    print("\nStep 3a: building stage1 (Debris vs. Arthropod)...")
    arthropod_source_classes = list(subclass_dest_names)
    if generic_arthropod_class:
        arthropod_source_classes.append(generic_arthropod_class)

    for split in SPLITS:
        debris_dest_dir = os.path.join(args.stage1_dest, split, debris_dest)
        n = populate_split_dir(class_splits[debris_dest][split], debris_dest_dir, args.copy)
        print(f"  {split}/{debris_dest}: {n} files")

        pooled = []
        for c in arthropod_source_classes:
            pooled.extend(class_splits[c][split])
        arthropod_dest = os.path.join(args.stage1_dest, split, STAGE1_ARTHROPOD_NAME)
        n = populate_split_dir(pooled, arthropod_dest, args.copy)
        print(f"  {split}/{STAGE1_ARTHROPOD_NAME} (pooled from {arthropod_source_classes}): {n} files")

    # --- Step 3: stage2 (subclasses + unidentified) ------------------------
    print("\nStep 3b: building stage2 (subclasses + unidentified)...")
    for dest_name in subclass_dest_names:
        for split in SPLITS:
            dest = os.path.join(args.stage2_dest, split, dest_name)
            n = populate_split_dir(class_splits[dest_name][split], dest, args.copy)
            print(f"  {split}/{dest_name}: {n} files")

    if generic_arthropod_class and stage2_unidentified_name:
        for split in SPLITS:
            dest = os.path.join(args.stage2_dest, split, stage2_unidentified_name)
            n = populate_split_dir(class_splits[generic_arthropod_class][split], dest, args.copy)
            print(f"  {split}/{stage2_unidentified_name} (from {generic_arthropod_class}): {n} files")

    if args.cleanup_rescaled:
        print(f"\nCleaning up intermediate rescaled directory: {args.rescaled_dir}")
        shutil.rmtree(args.rescaled_dir, ignore_errors=True)

    print("\nDone.")
    if not args.cleanup_rescaled:
        print(f"Rescaled crops:   {args.rescaled_dir}")
    print(f"Stage 1 dataset:  {args.stage1_dest}")
    print(f"Stage 2 dataset:  {args.stage2_dest}")


if __name__ == "__main__":
    main()