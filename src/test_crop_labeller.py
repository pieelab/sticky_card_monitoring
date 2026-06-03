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

def cli_args():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-i", "--id", type=str, dest="id_num", required=True,
                            help="Run ID number")
    args_parse.add_argument("-s", "--scans", type=str, dest="scans_dir", required=True,
                            help="Location where scanned sticky cards are stored")
    args_parse.add_argument("-a", "--annot_scans", type=str, dest="annot_scans_dir", required=True,
                            help="Location where annotated scanned cards are to be stored")
    args_parse.add_argument("-c", "--crops", type=str, dest="crops_dir", required=True,
                            help="Location where flatbug segments generated from sticky card scans are stored")
    args_parse.add_argument("-p", "--pt_path", type=str, dest="model_path",
                            help="Location of pretrained model")
    args = args_parse.parse_args()
    return vars(args)


def classify_segments(model_path, crops_dir, run_id, mappings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir = os.path.abspath(join(crops_dir, os.pardir))
    destination_crops_dir = join(root_dir, os.path.basename(crops_dir) + "_classified_test")

    if not os.path.exists(destination_crops_dir):
        os.makedirs(destination_crops_dir)

    pretrained = torch.load(model_path)
    pretrained.eval()
    print("Model loaded and set to eval mode")

    # TODO num_classes should not be hardcoded here
    segclass = SegmentClassifier(id = run_id, data_dir = crops_dir, num_classes = 5, device = device)
    segclass_loader = segclass.load_inference_data()
    sub = pd.DataFrame(columns=['category', 'id'])
    id_list = []
    pred_list = []
    # make subfolders for all classes
    for key,subfolder in mappings.items():
        sub_path = join(destination_crops_dir, subfolder)
        if not os.path.isdir(sub_path):
            os.mkdir(join(sub_path))
    pretrained = pretrained.to(device)
    with torch.no_grad():
        for image, idx in segclass_loader:
            image = {key: value.to(device) if hasattr(value, 'to') else value for key, value in image.items()}
            logits = pretrained(image)
            predicted = list(torch.argmax(logits, 1).cpu().numpy())
            probs = torch.nn.functional.softmax(logits, dim=1)
            for id, prediction, prob in zip(idx, predicted, probs):
                id_list.append(id)
                # experimenting with threshold
                # arth_classes = [1, 2, 3, 4] for 5-class model
                # if prediction.tolist() in arth_classes & probs[prediction.tolist()] < 0.7:
                #     pred_list.append(0)
                # else:
                #     pred_list.append(prediction.tolist())
                pred_list.append(prediction.tolist())
                shutil.copy(join(crops_dir, id),
                            join(destination_crops_dir, mappings[prediction]))
    sub = pd.DataFrame(columns=['category', 'id'])
    sub['category'] = pred_list
    sub['id'] = id_list
    sub['category'] = sub['category'].map({v: k for k, v in mappings.items()})
    sub = sub.sort_values(by='id')
    sub.to_csv(crops_dir + "inference.csv", index=False)
    return destination_crops_dir

def make_flag_list(crops_dir, parent, flag_list, card_set):
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
    # flatbug assigns 2 different cropnumbers. one is overall for each flatbug run, stored in the json
    # the other one restarts at 0 for every card and is stored in the filename
    # if the       json cropnumber id for card n's first annotation is m and for card n's xth annotation it is m+x
    # then the filename cropnumber id for card n's first annotation is 0 and for card n's xth annotation it is x
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
# json_data, flag_list, scans_dir_arg, annot_scans_dir, img_list):
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


def classify_prep(id_num, scans_dir, annot_scans_dir, crops_dir, model_path):
    Image.MAX_IMAGE_PIXELS = 933120000

    run_id = id_num + datetime.today().strftime("%m-%d-%H-%M")

    # these mappings are only applicable to 5-class model, hardcoded for convenience during testing
    # TODO modify OrigResNet50 and SizeResNet50 classes in label_crops.py to save class mappings as attribute
    mappings = {0: 'Arthropod', 1: 'Debris', 2: 'SWD_male', 3: 'SWD_parasitoid', 4: 'Small_black_weevil'}
    # 4-class mappings (from earlier versions):
    #mappings = {0: 'Other', 1: 'SWD_male', 2: 'SWD_parasitoid', 3: 'Weevil'}

    flag_list = defaultdict(list)
    # SWD_male_list = []
    # SWD_parasitoid_list = []
    # Small_black_weevil_list = []
    destination_dir = classify_segments(model_path, crops_dir, run_id, mappings)
    #annot_scans_dir = "C:\\Users\\ALANalysis\\flat-bug\\src\\2024_paul_abram_scans_annotated"
    if not os.path.isdir(join(annot_scans_dir, "pdfs")):
        os.mkdir(join(annot_scans_dir, "pdfs"))
    if not os.path.isdir(join(annot_scans_dir, "jpgs")):
        os.mkdir(join(annot_scans_dir, "jpgs"))
    scans_dir = "C:\\Users\\Public\\Documents\\scans\\2024_paul_abram_scans"
    json_dir = os.path.join(crops_dir, "coco_instances.json")
    #json_dir = "C:\\Users\\ALANalysis\\flat-bug\\src\\2024_paul_abram_crops\\coco_instances.json"

    card_set = set()
    img_list = dict()
    card_id = dict()
    colours = {"Small_black_weevil": 'r', "SWD_male": 'b', "SWD_parasitoid": 'g'}

    with open(json_dir, 'r') as file:
        json_data = json.load(file)

    for image in json_data["images"]:
        card_id[image["id"]] = Card(
            image["file_name"])  # use a card object so that we can store data about starting crop numbers later

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

# old, slower version:
# def make_flag_list(crops_dir_arg, parent_arg, flag_list_arg, card_set_arg):
#     card_match = r'[A-Z]*-[A-Z]*-[0-9]*'
#     for crop in os.listdir(crops_dir_arg):
#         if crop.endswith((".jpg",".png")):
#             if isfile(join(crops_dir_arg,crop)):
#                 match = re.compile(card_match)
#                 result = match.search(crop)
#                 #if result is not None: # some training data missing corresponding card
#                 flag_list_arg.append(Flag(crop, parent_arg, result.group(0)+".jpg"))
#                 card_set_arg.add(result.group(0))
#     return flag_list_arg, card_set_arg

# def annotate(json_data, flag_list_arg, scans_dir_arg, annot_scans_dir, img_list):
#     temp_annot_list = flag_list_arg
#     print(f"Starting annotation")
#     colours = {"Small_black_weevil": 'r', "SWD_male": 'b', "SWD_parasitoid": 'g'}
#     cropnum_match = r'CROPNUMBER_(\d*)_'
#     #for path, image in img_list:
#     for image in json_data["images"]:
#         start_id = 0
#         annot_flag = False
#         start_set = False
#         rect_list = []
#         for annot in temp_annot_list:
#             #if annot.card == path + ".jpg":
#             if annot.card == image["file_name"]: # if annotation is for current card
#                 print(annot.card)
#                 print(annot.crop)
#                 annot_flag = True
#                 match = re.compile(cropnum_match)
#                 annot_id = match.search(annot.crop).group(1)
#                 for annotation in json_data["annotations"]:
#                     if annotation["image_id"] == image["id"] and start_set == False:
#                         start_id = annotation["id"]
#                         start_set = True
#                     if start_set:
#                         if annotation["id"] == (start_id + int(annot_id)):
#                             annot.bbox = annotation["bbox"]
#                             rect_list.append(patches.Rectangle((annot.bbox[0], annot.bbox[1]), annot.bbox[2], annot.bbox[3],
#                                                                edgecolor=colours[annot.parent], facecolor='none',
#                                                                lw=0.5))
#                             json_data["annotations"].remove(annotation)
#                 temp_annot_list.remove(annot)
#             if annot_flag: # if any annotations were found for this card
#                 #img = Image.open(glob.glob(join(scans_dir_arg, image["file_name"][:-4]+"*"))[0])
#
#                 fig, ax = plt.subplots()
#                 #ax.imshow(img)
#                 ax.imshow(img_list[annot.card])
#                 for rect in rect_list:
#                     new_r = copy(rect)
#                     ax.add_patch(new_r)
#                 no_extension = re.compile(r'(.*)\.jpg').search(image["file_name"]).group(1)
#                 plt.axis('off')
#                 plt.savefig(join(annot_scans_dir, "pdfs", no_extension + "-annotated.pdf"), dpi=2400, bbox_inches="tight")
#                 plt.close()
#                 pages = convert_from_path(join(annot_scans_dir, "pdfs", no_extension + "-annotated.pdf"), dpi=2400)
#                 for count, page in enumerate(pages):
#                     page.save(join(annot_scans_dir, "jpgs", no_extension + "-annotated.jpg"), 'JPEG')
#     print("Saved annotated image")
