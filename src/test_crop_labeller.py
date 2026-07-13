import re, torch, os, sys, glob, gc, json, argparse, shutil
from os.path import join
from pathlib import Path
from PIL import Image
from datetime import datetime
import torchvision.transforms.v2 as transforms  # composable transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import ultralytics
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from os.path import isfile, join
from pdf2image import convert_from_path
from copy import copy
from collections import defaultdict
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import patches
matplotlib.use('Agg')
torch.set_printoptions(sci_mode=False)
gc.collect()
from label_crops import ImageFolderWithPaths, SegmentClassifier
from utils.Flag import Flag
from utils.Card import Card

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.ToImage(),
    transforms.Resize(size=(224, 224), antialias=True),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def cli_args():
    """
    Parse command-line arguments for the classification and annotation pipeline.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - id_num (str): Run ID prefix used to construct a unique run identifier.
        - scans_dir (str): Path to the directory containing scanned sticky card images.
        - annot_scans_dir (str): Path to the directory where annotated card images will be saved.
        - crops_dir (str): Path to the directory containing flatbug-segmented crop images.
        - stage1_model_path (str) : Path to the stage 1 model state dict (.pt file)
        - stage2_model_path (str) : Path to the stage 2 model state dict (.pt file)
        - stage1_arch (str): Architecture for stage 1 model (default: resnet50)
        - stage2_arch (str): Architecture for stage 2 model (default: resnet50)
        - stage1_size_aware (bool): Whether stage 1 model is size-aware
        - stage2_size_aware (bool): Whether stage 2 model is size-aware
    """
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-i", "--id", type=str, dest="id_num", required=True,
                            help="Run ID number")
    args_parse.add_argument("-s", "--scans", type=str, dest="scans_dir", required=True,
                            help="Location where scanned sticky cards are stored")
    args_parse.add_argument("-a", "--annot_scans", type=str, dest="annot_scans_dir", required=True,
                            help="Location where annotated scanned cards are to be stored")
    args_parse.add_argument("-c", "--crops", type=str, dest="crops_dir", required=True,
                            help="Location where flatbug segments generated from sticky card scans are stored")
    args_parse.add_argument("-p1", "--stage1_pt_path", type=str, dest="stage1_model_path", required=True,
                            help="Location of the stage 1 model state dict")
    args_parse.add_argument("-p2", "--stage2_pt_path", type=str, dest="stage2_model_path", required=True,
                            help="Location of the stage 2 model state dict")
    args_parse.add_argument("--stage1_arch", type=str, dest="stage1_arch", default="resnet50",
                            help="Architecture for stage 1 model (resnet50 or dinov2 variant)")
    args_parse.add_argument("--stage2_arch", type=str, dest="stage2_arch", default="resnet50",
                            help="Architecture for stage 2 model (resnet50 or dinov2 variant)")
    args_parse.add_argument("--stage1_size_aware", dest="stage1_size_aware", action='store_true',
                            help="Whether stage 1 model is size-aware")
    args_parse.add_argument("--stage2_size_aware", dest="stage2_size_aware", action='store_true',
                            help="Whether stage 2 model is size-aware")

    args = args_parse.parse_args()
    return vars(args)

class FilteredInferenceDataset(Dataset):
    """
    Loads a specific list of image files from a directory for inference.

    Used for stage 2, which only needs to run on the subset of crops that 
    stage 1 predicted as "Arthropod" rather than every file in crops_dir.

    Attributes
    ----------
    crops_dir : str
        Directory the files live in.
    filenames : list of str
        Filenames (relative to crops_dir) to load
    size_aware : bool
        Whether to include dummy npb (number of pixels before resizing) values
        for size-aware models.
    """

    def __init__(self, crops_dir, filenames, size_aware=False):
        self.crops_dir = crops_dir
        self.filenames = filenames
        self.size_aware = size_aware

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(join(self.crops_dir, filename)).convert("RGB")
        img = INFERENCE_TRANSFORM(img)
        
        result = {"img": img}
        
        # If model is size-aware, provide a dummy normalized npb value
        # (actual npb values aren't available during inference with this dataset)
        if self.size_aware:
            result["npb"] = torch.tensor(0.0, dtype=torch.float32)
        
        return result, filename
    
def run_model_on_loader(model, loader, device):
    """
    Runs a loaded model over a DataLoader and collects predictions.

    Parameters
    ----------
    model : torch.nn.Module
        Model in eval mode.
    loader : torch.utils.data.DataLoader
        Yields (input_dict, filename) batches.
    device : torch.device

    Returns
    -------
    dict
        Mapping from filename (str) to a predicted class index (int)
    """

    predictions = {}
    with torch.no_grad():
        for inputs, filenames in loader:
            inputs = {key: value.to(device) if hasattr(value, "to") else value
                      for key, value in inputs.items()}
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for filename, pred in zip(filenames, preds):
                predictions[filename] = int(pred)
    return predictions


def load_model_from_state_dict(state_dict_path, num_classes, architecture, size_aware, device):
    """
    Load a model from a saved state dict.

    Reconstructs the model architecture and loads the state dict weights.

    Parameters
    ----------
    state_dict_path : str
        Path to the saved state dict (.pt file)
    num_classes : int
        Number of classes the model was trained on
    architecture : str
        Model architecture ('resnet50' or dinov2 variant like 'dinov2_vitb14')
    size_aware : bool
        Whether the model was trained with size awareness
    device : torch.device
        Device to load the model onto

    Returns
    -------
    model : torch.nn.Module
        Model with loaded state dict, in eval mode
    """
    print(f"Loading {architecture} model (size_aware={size_aware}) from state dict")
    
    # Create a temporary SegmentClassifier to build the model
    # Using dummy values for parameters that aren't needed for inference
    temp_classifier = SegmentClassifier(
        id="inference",
        data_dir=".",  # Dummy, not used for inference
        num_classes=num_classes,
        device=device,
        mean_npb=1.0 if size_aware else None,  # Dummy value, not actually used for inference
        std_npb=1.0 if size_aware else None,
    )
    
    # Build the model architecture
    temp_classifier.load_model(backbone=architecture)
    model = temp_classifier.model
    
    # Load the state dict
    try:
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
        print(f"Successfully loaded state dict from {state_dict_path}")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        raise
    
    model.eval()
    return model


def classify_segments_hierarchical(stage1_model_path, stage2_model_path, crops_dir, run_id, 
                                   stage1_mappings, stage2_mappings, arthropod_class_name,
                                   stage1_arch, stage2_arch, stage1_size_aware, stage2_size_aware,
                                   batch_size=32, num_workers=4):
    """
    Run two-stage hierarchical inference on segmented crop images and copy them
    into class-labelled subdirectories.
 
    Stage 1 classifies every crop as Debris or Arthropod. Stage 2 then runs only
    on crops predicted Arthropod, resolving them into a specific subclass. Every
    crop ends up copied into exactly one final class subfolder: either the
    Debris folder, or one of the stage 2 subclass folders.
 
    Parameters
    ----------
    stage1_model_path : str
        Path to the saved stage 1 model state dict (.pt file).
    stage2_model_path : str
        Path to the saved stage 2 model state dict (.pt file).
    crops_dir : str
        Path to the directory containing crop images to classify.
    run_id : str
        Unique run identifier passed to SegmentClassifier for data loading.
    stage1_mappings : dict
        Mapping from stage 1 class index to label, e.g.
        {0: 'Arthropod', 1: 'Debris'}.
    stage2_mappings : dict
        Mapping from stage 2 class index to label, e.g.
        {0: 'SWD_male', 1: 'SWD_parasitoid', 2: 'Small_black_weevil',
         3: 'Unidentified_Arthropod'}.
    arthropod_class_name : str
        The label in stage1_mappings that means "send to stage 2", e.g.
        'Arthropod'.
    stage1_arch : str
        Architecture for stage 1 model ('resnet50' or dinov2 variant).
    stage2_arch : str
        Architecture for stage 2 model ('resnet50' or dinov2 variant).
    stage1_size_aware : bool
        Whether stage 1 model is size-aware.
    stage2_size_aware : bool
        Whether stage 2 model is size-aware.
    batch_size : int, default=32
        Batch size for both stages' DataLoaders.
    num_workers : int, default=4
        Number of DataLoader worker processes.
 
    Returns
    -------
    destination_crops_dir : str
        Path to the newly created directory containing crops sorted into
        class-labelled subdirectories (Debris + every stage 2 subclass).
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = os.path.abspath(join(crops_dir, os.pardir))
    destination_crops_dir = join(root_dir, os.path.basename(crops_dir) + "_classified_test")

    if not os.path.exists(destination_crops_dir):
        os.makedirs(destination_crops_dir)

    final_labels = [v for v in stage1_mappings.values() if v != arthropod_class_name] + list(stage2_mappings.values())

    for label in final_labels:
        sub_path = join(destination_crops_dir, label)
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)

    print("Loading stage 1 (binary) model")
    stage1_model = load_model_from_state_dict(
        stage1_model_path,
        num_classes=len(stage1_mappings),
        architecture=stage1_arch,
        size_aware=stage1_size_aware,
        device=device
    )

    stage1_helper = SegmentClassifier(id=run_id, data_dir=crops_dir,
                                      num_classes=len(stage1_mappings), device=device)
    stage1_loader = stage1_helper.load_inference_data()

    print("Running stage 1 inference")
    stage1_preds = run_model_on_loader(stage1_model, stage1_loader, device)
    stage1_labels = {filename: stage1_mappings[idx] for filename, idx in stage1_preds.items()}

    del stage1_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    arthropod_filenames = [f for f, label in stage1_labels.items() if label == arthropod_class_name]

    print(f"Loading stage 2 (subclass) - {len(arthropod_filenames)} crops to classify")

    stage2_model = load_model_from_state_dict(
        stage2_model_path,
        num_classes=len(stage2_mappings),
        architecture=stage2_arch,
        size_aware=stage2_size_aware,
        device=device
    )

    stage2_preds = {}
    if len(arthropod_filenames) > 0:
        stage2_dataset = FilteredInferenceDataset(crops_dir, arthropod_filenames, size_aware=stage2_size_aware)
        stage2_loader = DataLoader(stage2_dataset, batch_size=batch_size, num_workers=num_workers)
        stage2_preds = run_model_on_loader(stage2_model, stage2_loader, device)

    del stage2_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    id_list = []
    category_list = []

    for filename, stage1_label in stage1_labels.items():
        if stage1_label == arthropod_class_name:
            final_label = stage2_mappings[stage2_preds[filename]]
        else:
            final_label = stage1_label
        id_list.append(filename)
        category_list.append(final_label)
        shutil.copy(join(crops_dir, filename), join(destination_crops_dir, final_label))

    sub = pd.DataFrame({"id": id_list, "category": category_list})
    sub = sub.sort_values(by="id")
    sub.to_csv(join(crops_dir, "inference.csv"), index=False)
    return destination_crops_dir

def make_flag_list(crops_dir, parent, flag_list, card_set):
    """
    Scan a directory of classified crop images and populate a flag list grouped by card.

    For each image file in crops_dir, extracts the parent card identifier from the
    filename using a regex pattern, creates a Flag object, and appends it to the
    flag_list under that card's key. Also records the card ID in card_set.

    Parameters
    ----------
    crops_dir : str
        Path to the directory of classified crop images (one insect class).
    parent : str
        The class label for all crops in this directory (e.g. 'SWD_male').
        Stored as an attribute on each Flag object.
    flag_list : collections.defaultdict of list
        Mapping from card filename (str) to a list of Flag objects. Updated in place.
    card_set : set
        Set of card ID strings (without extension). Updated in place.

    Returns
    -------
    flag_list : collections.defaultdict of list
        Updated mapping of card filenames to their associated Flag objects.
    card_set : set
        Updated set of card ID strings found in crops_dir.
    """
    card_match = r'[A-Z]*-[A-Z]*-[0-9]*'
    for crop in os.listdir(crops_dir):
        if crop.endswith((".jpg",".png")) & isfile(join(crops_dir,crop)):
            match = re.compile(card_match)
            result = match.search(crop)
            #if result is not None: # some training data missing corresponding card
            flag_list[result.group(0)+".jpg"].append(Flag(crop, parent))
            card_set.add(result.group(0))
    return flag_list, card_set

def add_json_data(flag_list, card_id, json_data, colours):
    """
    Match COCO annotations to Flag objects and build a dict of matplotlib Rectangle patches.

    Iterates over all annotations in a COCO-format JSON, determines the starting
    annotation ID for each card, and matches each annotation to its corresponding
    Flag object using the crop number embedded in the crop filename. Assigns the
    bounding box to matched flags and creates a Rectangle patch for each.

    Parameters
    ----------
    flag_list : collections.defaultdict of list
        Mapping from card filename (str) to a list of Flag objects, as returned
        by make_flag_list.
    card_id : dict
        Mapping from COCO image ID (int) to Card objects. Card objects are
        updated in place with their crop_start_id.
    json_data : dict
        Parsed COCO-format JSON containing at minimum "annotations" and "images" keys.
    colours : dict
        Mapping from class label (str) to a matplotlib colour string, used to
        colour the bounding box rectangles. e.g. {'SWD_male': 'b', ...}

    Returns
    -------
    flag_list : collections.defaultdict of list
        Updated flag list with bounding boxes assigned to matched Flag objects.
    card_id : dict
        Updated card dict with crop_start_id set on each Card object.
    rect_list : collections.defaultdict of list
        Mapping from Card objects to lists of matplotlib Rectangle patches,
        ready for rendering onto the card image.
    """
    start_set = False
    start_id = 0
    prev_card_id = None
    rect_list = defaultdict(list)
    # go thru all annotations in json file
    cropnum_match = r'CROPNUMBER_(\d*)_'
    match = re.compile(cropnum_match)
    # for each annotation
    for annot in json_data["annotations"]:
        # each time you move on to annotations pertaining to a new image, record the id of the first annotation
        if annot["image_id"] != prev_card_id:
            card_id[annot["image_id"]].crop_start_id = annot["id"]
            #start_id = annot["id"]
            #start_set = True
        # for the subset of flag objects related to this annotation's cards
        for flag in flag_list[card_id[annot["image_id"]].filename]:
            flag_id = match.search(flag.crop).group(1)
            # if you've found the corresponding flag object to the current annotation
            if annot["id"] == (card_id[annot["image_id"]].crop_start_id + int(flag_id)):
                # add bounding box to annotation object
                flag.bbox = annot["bbox"]
                rect_list[card_id[annot["image_id"]]].append(patches.Rectangle((flag.bbox[0], flag.bbox[1]), flag.bbox[2], flag.bbox[3],
                                                   edgecolor=colours[flag.parent], facecolor='none',
                                                   lw=0.5))
        prev_card_id = annot["image_id"]
    return flag_list, card_id, rect_list


def annotate_card(card, rects, scans_dir, annot_scans_dir):
    """
    Render bounding box annotations onto a scanned sticky card image and save outputs.

    Opens the original scan, overlays all bounding box Rectangle patches, and saves
    the result as both a high-resolution PDF and a JPEG to their respective subdirectories
    within annot_scans_dir.

    Parameters
    ----------
    card : Card
        Card object whose filename attribute identifies the scan to annotate.
    rects : list of matplotlib.patches.Rectangle
        Bounding box patches to overlay on the card image.
    scans_dir : str
        Path to the directory containing the original scanned card images.
    annot_scans_dir : str
        Path to the output directory. Must already contain pdfs/ and jpgs/
        subdirectories (created by classify_prep).

    Returns
    -------
    None
    """
    print(f"Annotating card {card.filename}")
    img = Image.open(join(scans_dir, card.filename))
    fig, ax = plt.subplots()
    ax.imshow(img)
    for rect in rects:
        new_r = copy(rect)
        ax.add_patch(new_r)
    no_extension = re.compile(r'(.*)\.jpg').search(card.filename).group(1)
    plt.axis('off')
    plt.savefig(join(annot_scans_dir, "pdfs", no_extension + "-annotated.pdf"), dpi=2400, bbox_inches="tight")
    plt.close()
    pages = convert_from_path(join(annot_scans_dir, "pdfs", no_extension + "-annotated.pdf"), dpi=2400)
    for count, page in enumerate(pages):
        page.save(join(annot_scans_dir, "jpgs", no_extension + "-annotated.jpg"), 'JPEG')
    print("Saved annotated image")


def classify_prep(id_num, scans_dir, annot_scans_dir, crops_dir, stage1_model_path, stage2_model_path,
                  stage1_arch, stage2_arch, stage1_size_aware, stage2_size_aware):
    """
    Orchestrate the full hierarchical classification and annotation pipeline for
    sticky card scans.
 
    Runs two-stage segment classification (Debris/Arthropod, then subclass),
    builds flag and card data structures from COCO JSON output, matches
    annotations to classified crops, and produces annotated card images with
    colour-coded bounding boxes for each insect class of interest.

    Parameters
    ----------
    id_num : str
        Run ID prefix combined with the current datetime to form a unique run ID.
    scans_dir : str
        Path to the directory containing original scanned sticky card images.
    annot_scans_dir : str
        Path to the directory where annotated outputs (PDFs and JPEGs) will be saved.
        pdfs/ and jpgs/ subdirectories are created here if they do not exist.
    crops_dir : str
        Path to the directory containing flatbug crop images and the
        coco_instances.json annotation file.
    stage1_model_path : str
        Path to the stage 1 model state dict (.pt).
    stage2_model_path : str
        Path to the stage 2 model state dict (.pt).
    stage1_arch : str
        Architecture for stage 1 model.
    stage2_arch : str
        Architecture for stage 2 model.
    stage1_size_aware : bool
        Whether stage 1 model is size-aware.
    stage2_size_aware : bool
        Whether stage 2 model is size-aware.
    """
    Image.MAX_IMAGE_PIXELS = 933120000

    run_id = id_num + datetime.today().strftime("%m-%d-%H-%M")

    # Stage mappings - MUST match the order from ImageFolder's alphabetical class_to_idx
    # from each stage's training run
    stage1_mappings = {0: 'Arthropod', 1: 'Debris'}
    stage2_mappings = {0: 'SWD_male', 1: 'SWD_parasitoid', 2: 'Small_black_weevil', 3: 'Unidentified_Arthropod'}
    arthropod_class_name = 'Arthropod'
 
    flag_list = defaultdict(list)
    destination_dir = classify_segments_hierarchical(
        stage1_model_path, stage2_model_path, crops_dir, run_id,
        stage1_mappings, stage2_mappings, arthropod_class_name,
        stage1_arch, stage2_arch, stage1_size_aware, stage2_size_aware
    )

    if not os.path.isdir(join(annot_scans_dir, "pdfs")):
        os.mkdir(join(annot_scans_dir, "pdfs"))
    if not os.path.isdir(join(annot_scans_dir, "jpgs")):
        os.mkdir(join(annot_scans_dir, "jpgs"))

    json_dir = os.path.join(crops_dir, "coco_instances.json")

    card_set = set()
    img_list = dict()
    card_id = dict()
    colours = {"Small_black_weevil": 'r', "SWD_male": 'b', "SWD_parasitoid": 'g'}

    with open(json_dir, 'r') as file:
        json_data = json.load(file)

    for image in json_data["images"]:
        card_id[image["id"]] = Card(image["file_name"])

    flag_list, card_set = make_flag_list(join(destination_dir, 'Small_black_weevil'), "Small_black_weevil",
                                         flag_list, card_set)
    flag_list, card_set = make_flag_list(join(destination_dir, 'SWD_parasitoid'), "SWD_parasitoid", flag_list,
                                         card_set)
    flag_list, card_set = make_flag_list(join(destination_dir, 'SWD_male'), "SWD_male", flag_list, card_set)
    flag_list, card_id, rect_list = add_json_data(flag_list, card_id, json_data, colours)
    for card, rects in rect_list.items():
        annotate_card(card, rects, scans_dir, annot_scans_dir)


def main():
    classify_prep(**cli_args())

if __name__ == '__main__':
    main()