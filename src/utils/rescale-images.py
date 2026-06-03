import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# d'après Melika Baghooee https://github.com/darsa-group/size-aware-classification/blob/main/04_rescale-images.py

SRC_DIR = "C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training_unsplit"
TAR_DIR = "C:\\Users\\ALANalysis\\flat-bug\\src\\A_rubi_training_unsplit_rescaled"
TARGET_SIZE = 224

BLACK = [0, 0, 0]

def count_mask_pixels(image):
    # Convert image to RGB if it's in BGR
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    num_pixels = np.sum(np.all(image != [0, 0, 0], axis=-1))

    return num_pixels


def pad_scale(im, tar):
    h, w = im.shape[:2]

    if w > h:
        vertical_margin = (w - h) / 2
        top_margin = np.ceil(vertical_margin).astype(int)
        bottom_margin = np.floor(vertical_margin).astype(int)
        pad = cv2.copyMakeBorder(im, top_margin, bottom_margin, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    else:
        horizontal_margin = (h - w) / 2
        left_margin = np.ceil(horizontal_margin).astype(int)
        right_margin = np.floor(horizontal_margin).astype(int)
        pad = cv2.copyMakeBorder(im, 0, 0, left_margin, right_margin, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(pad, (tar, tar))
    resized_mask_pixels = count_mask_pixels(resized_image)
    return resized_image, im.shape[0:2], resized_mask_pixels


for species_folder in os.listdir(SRC_DIR):
    species_folder_path = os.path.join(SRC_DIR, species_folder)

    if os.path.isdir(species_folder_path):
        images = glob(os.path.join(species_folder_path, '*.png'))
        # target_folder_path = os.path.join(TAR_DIR, genus_folder, species_folder)
        target_folder_path = os.path.join(TAR_DIR, species_folder)
        os.makedirs(target_folder_path, exist_ok=True)

        # Process each image in the species folder
        for item_path in tqdm(images, position=0, leave=True):
            item = os.path.basename(item_path)
            img = cv2.imread(item_path)

            if img is None:
                print(f"Error reading image {item_path}")
                continue

            h, w, _ = img.shape

            # Rotate if necessary so longer side is horizontal
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h, w = img.shape[:2]

            mask_pixels = count_mask_pixels(img)
            resized_image, original_size, resized_mask_pixels = pad_scale(img, TARGET_SIZE)

            # Save resized and rotated image to target directory
            file_name, file_extension = os.path.splitext(item)
            new_file_name = f"{file_name}_{w}_{h}_{mask_pixels}_{resized_mask_pixels}{file_extension}"
            target_item_path = os.path.join(target_folder_path, new_file_name)

            cv2.imwrite(target_item_path, resized_image)

# Count saved images
saved_images = glob(os.path.join(TAR_DIR, '*', '*.png'))
print(f"Total saved images: {len(saved_images)}")
