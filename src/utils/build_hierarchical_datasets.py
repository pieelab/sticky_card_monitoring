"""
Derives two new dataset directory trees from an already-split 5-class
dataset, for training a hierarchical Debris/Arthropod -> subclass model
pair.

Given a source directory with the structure:

    source_split_dir/
        train/
            Debris/
            Arthropod/          (generic / unidentified arthropod)
            ClassA/
            ClassB/
            ClassC/
        val/
            ...
        test/
            ...

this produces:

    stage1_dest/                (Debris vs. Arthropod)
        train/
            Debris/
            Arthropod/           <- pooled from Arthropod + ClassA + ClassB + ClassC
        val/  ...
        test/ ...

    stage2_dest/                 (4-way: subclasses + unidentified arthropod)
        train/
            ClassA/
            ClassB/
            ClassC/
            Unidentified_Arthropod/   <- pooled from the generic Arthropod class
        val/  ...
        test/ ...

Stage 2 is NOT subclasses-only: the generic/unidentified arthropod class
is a genuine 4th output class here (crops that are clearly arthropods but
don't confidently match one of the specific subclasses), not something
to discard after stage 1. Stage 1 only decides Debris vs. "is this an
arthropod at all"; stage 2 decides which kind, including "unknown kind".

Train/val/test membership is preserved exactly as in the source split
(files are not re-shuffled), so the two stages are evaluated on
consistent data.

By default files are hard-linked where possible (falls back to copy
across filesystems) to avoid duplicating disk space; pass --copy to
force plain copies instead.

Usage
-----
python build_hierarchical_datasets.py \
    --source-split-dir /path/to/existing/5class/split \
    --stage1-dest /path/to/stage1_data \
    --stage2-dest /path/to/stage2_data \
    --debris-class Debris \
    --generic-arthropod-class Arthropod \
    --subclasses ClassA,ClassB,ClassC \
    --stage2-unidentified-name Unidentified_Arthropod
"""

import argparse
import os
import shutil

from tqdm import tqdm

SPLITS = ("train", "val", "test")


def cli_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source-split-dir", required=True,
                         help="Root of the existing split dataset "
                              "(contains train/val/test subfolders)")
    parser.add_argument("--stage1-dest", required=True,
                         help="Output root for the binary Debris/Arthropod dataset")
    parser.add_argument("--stage2-dest", required=True,
                         help="Output root for the subclass-only dataset")
    parser.add_argument("--debris-class", default="Debris",
                         help="Folder name of the debris class in the source split")
    parser.add_argument("--generic-arthropod-class", default="Arthropod",
                         help="Folder name of the generic/unidentified arthropod "
                              "class in the source split. Pass '' if you don't "
                              "have one and stage 1's Arthropod class should be "
                              "made up of the subclasses only.")
    parser.add_argument("--subclasses", required=True,
                         help="Comma-separated folder names of the specific "
                              "arthropod subclasses, e.g. 'ClassA,ClassB,ClassC'")
    parser.add_argument("--stage2-unidentified-name", default="Unidentified_Arthropod",
                         help="Destination folder name in stage2_dest for the "
                              "pooled generic/unidentified arthropod class. This "
                              "becomes stage 2's 4th output class. Pass '' to "
                              "exclude it (subclasses-only stage 2) if you don't "
                              "want an unidentified-arthropod output.")
    parser.add_argument("--copy", action="store_true",
                         help="Force plain file copies instead of hard links")
    return parser.parse_args()


def link_or_copy(src, dst, force_copy=False):
    """
    Hard-links src to dst, falling back to a copy if hard-linking isn't
    possible (e.g. different filesystems) or if force_copy is set.
    """
    if os.path.exists(dst):
        return
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def populate_class_dir(source_class_dirs, dest_dir, force_copy):
    """
    Links/copies every file from one or more source class directories into
    a single destination class directory.

    Parameters
    ----------
    source_class_dirs : list of str
        Paths to source folders whose contents should be pooled together.
    dest_dir : str
        Destination folder (created if it doesn't exist).
    force_copy : bool
        Passed through to link_or_copy.

    Returns
    -------
    int
        Number of files written.
    """
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    for source_dir in source_class_dirs:
        if not os.path.isdir(source_dir):
            continue
        for filename in os.listdir(source_dir):
            src_path = os.path.join(source_dir, filename)
            if not os.path.isfile(src_path):
                continue
            dst_path = os.path.join(dest_dir, filename)
            link_or_copy(src_path, dst_path, force_copy=force_copy)
            count += 1
    return count


def build_stage1(source_split_dir, stage1_dest, debris_class,
                  generic_arthropod_class, subclasses, force_copy):
    """
    Builds the binary Debris-vs-Arthropod dataset tree.

    The Arthropod class in the output is the pool of the generic
    arthropod class (if given) plus every subclass. The Debris class is
    copied as-is.
    """
    print("Building stage 1 (Debris vs. Arthropod) dataset...")
    arthropod_source_names = list(subclasses)
    if generic_arthropod_class:
        arthropod_source_names.append(generic_arthropod_class)

    for split in tqdm(SPLITS, desc="Stage 1 splits"):
        split_src = os.path.join(source_split_dir, split)

        debris_src = [os.path.join(split_src, debris_class)]
        debris_dst = os.path.join(stage1_dest, split, debris_class)
        n_debris = populate_class_dir(debris_src, debris_dst, force_copy)

        arthropod_src_dirs = [os.path.join(split_src, c) for c in arthropod_source_names]
        arthropod_dst = os.path.join(stage1_dest, split, "Arthropod")
        n_arthropod = populate_class_dir(arthropod_src_dirs, arthropod_dst, force_copy)

        print(f"  [{split}] Debris: {n_debris} files | Arthropod (pooled): {n_arthropod} files")


def build_stage2(source_split_dir, stage2_dest, subclasses, generic_arthropod_class,
                  stage2_unidentified_name, force_copy):
    """
    Builds the stage 2 dataset tree: the specific subclasses, plus (unless
    disabled) a pooled "unidentified arthropod" class built from the
    generic Arthropod folder.
    """
    print("Building stage 2 (subclass) dataset...")
    for split in tqdm(SPLITS, desc="Stage 2 splits"):
        split_src = os.path.join(source_split_dir, split)
        for subclass in subclasses:
            src_dir = os.path.join(split_src, subclass)
            dst_dir = os.path.join(stage2_dest, split, subclass)
            n = populate_class_dir([src_dir], dst_dir, force_copy)
            print(f"  [{split}] {subclass}: {n} files")

        if generic_arthropod_class and stage2_unidentified_name:
            src_dir = os.path.join(split_src, generic_arthropod_class)
            dst_dir = os.path.join(stage2_dest, split, stage2_unidentified_name)
            n = populate_class_dir([src_dir], dst_dir, force_copy)
            print(f"  [{split}] {stage2_unidentified_name} (from {generic_arthropod_class}): {n} files")


def main():
    args = cli_args()
    subclasses = [c.strip() for c in args.subclasses.split(",") if c.strip()]
    generic_arthropod_class = args.generic_arthropod_class.strip() or None

    for split in SPLITS:
        split_dir = os.path.join(args.source_split_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected split directory not found: {split_dir}\n"
                f"--source-split-dir should point at the root containing "
                f"train/val/test subfolders."
            )

    build_stage1(
        args.source_split_dir, args.stage1_dest, args.debris_class,
        generic_arthropod_class, subclasses, args.copy,
    )
    build_stage2(
        args.source_split_dir, args.stage2_dest, subclasses,
        generic_arthropod_class, args.stage2_unidentified_name.strip() or None,
        args.copy,
    )

    print("\nDone.")
    print(f"Stage 1 (binary) dataset: {args.stage1_dest}")
    print(f"Stage 2 (subclass) dataset: {args.stage2_dest}")


if __name__ == "__main__":
    main()