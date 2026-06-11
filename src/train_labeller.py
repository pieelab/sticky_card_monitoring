import argparse
import random
import os
import shutil
from datetime import datetime
from math import floor, ceil
from collections import defaultdict
from label_crops import SegmentClassifier, calculate_mean_std_npb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import ultralytics
import torch
import torchvision.transforms.v2 as transforms  # composable transforms
from torchvision.transforms import RandomRotation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def cli_args():
    """
    Function to accept command line arguments

    Arguments
    ---------
    -i, --id : str
        Run ID number
    -s, --source : str
        Unsplit dataset location
    -d, --destination : str
        Location where split dataset should be stored
    -m, --mode : str
        Whether to train a model from scratch
    -p, --pt_path : str
        Location of pretrained model
    -l, --split
        Flag whether to split the data or not
    -r, --arch : str
        Model architecture ("resnet", or dinov2 variant 
        (eg. "dinov2_vitb14"))
    -a, --size_aware
        Whether to perform size-aware classification
    -e, --epochs
        Number of epochs to train over

    Returns
    -------
    vars(args) : dict
        Dictionary with all arguments and flags.

    """
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-i", "--id", type=str, dest="id_num", required=True,
                            help="Run ID number")
    args_parse.add_argument("-s", "--source", type=str, dest="source_data_dir",
                            help="Unsplit dataset location")
    args_parse.add_argument("-d", "--destination", type=str, dest="destination_data_dir", required=True,
                            help="Location where split dataset should be stored")
    args_parse.add_argument("-m", "--mode", type=str, dest="mode", required=True,
                            help="Whether to train model from scratch")
    args_parse.add_argument("-p", "--pt_path", type=str, dest="model_path",
                            help="Location of pretrained model")
    args_parse.add_argument("-l", "--split", dest="split", action='store_true',
                            help="Whether to split dataset")
    args_parse.add_argument("-r", "--arch", type=str, dest="arch",
                            help="Model architecture (resnet, bioclip)")
    args_parse.add_argument("-a", "--size_aware", dest="size_aware", action='store_true',
                            help="Whether to perform size-aware classification")
    args_parse.add_argument("-e", "--epochs", type=int, dest="epochs", required=True,
                            help="Number of epochs")
    args = args_parse.parse_args()
    return vars(args)


def plot_training_history(history, run_id):
    """
    Plots the training and validation loss and accuracy.

    Saves plots to output/train directory

    Parameters
    ----------
    history : dict
        Contains past train and validation losses
    run_id : int
        Run identification number
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    #ax1.plot(history['train_losses'], label='Train Loss')
    #ax1.plot(history['val_losses'], label='Validation Loss')
    for i in range(len(history['train_losses'])):
        ax1.plot([pt[i] for pt in history['train_losses']],label = 'id %s'%i)
        ax1.plot([pt[i] for pt in history['val_losses']], label='id %s' % i)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "train", f"training_history_{run_id}.png"))

def split_data(source_data_dir, destination_data_dir):
    """
    Function to create (80|10|10) train, validation, and test splits.

    Parameters
    ----------
    source_data_dir : str
        Path to the unsplit data directory
    destination_data_dir : str
        Path to the split data directory

    Returns
    -------
    filenames : collections.defaultdict
        A defaultdict mapping each label (str) to a shuffled list of
        filenames (list of str) belonging to that label.
    num_files : int
        Total number of files found across all labels in source_data_dir.

    Raises
    ------
    FileNotFoundError
        If source_data_dir or destination_data_dir does not exist.
    """
    filenames = defaultdict(list)
    num_files = 0
    with os.scandir(source_data_dir) as ents:
        for e in ents:
            if e.is_dir():
                for f in os.scandir(e.path):
                    filenames[e.name].append(f.name)
                    num_files = num_files + 1
    train_files = open(os.path.join(destination_data_dir, "train_filenames_temp.csv"), "w")
    train_files.truncate(0)
    test_files = open(os.path.join(destination_data_dir, "test_filenames_temp.csv"), "w")
    test_files.truncate(0)
    val_files = open(os.path.join(destination_data_dir, "val_filenames_temp.csv"), "w")
    val_files.truncate(0)
    pbar = tqdm(total=num_files, desc="Copying files", dynamic_ncols=True, unit="files", position=0, leave=True)
    for label in filenames:
        random.shuffle(filenames[label])
        shutil.rmtree(os.path.join(destination_data_dir, "train", label))
        os.makedirs(os.path.join(destination_data_dir, "train", label))
        shutil.rmtree(os.path.join(destination_data_dir, "val", label))
        os.makedirs(os.path.join(destination_data_dir, "val", label))
        shutil.rmtree(os.path.join(destination_data_dir, "test", label))
        os.makedirs(os.path.join(destination_data_dir, "test", label))
        i = 0
        for f in filenames[label]:
            if i < floor(0.8 * len(filenames[label])):
                train_files.write(f + "," + label + "\n")
                shutil.copy(os.path.join(source_data_dir, label, f),
                            os.path.join(destination_data_dir, "train", label))
            elif i < floor(0.9 * len(filenames[label])):
                test_files.write(f + "," + label + "\n")
                shutil.copy(os.path.join(source_data_dir, label, f),
                            os.path.join(destination_data_dir, "val", label))
            elif i < floor(len(filenames[label])):
                test_files.write(f + "," + label + "\n")
                shutil.copy(os.path.join(source_data_dir, label, f),
                            os.path.join(destination_data_dir, "test", label))
            i = i + 1
            pbar.update(1)
    train_files.close()
    val_files.close()
    test_files.close()
    return filenames, num_files

def classify(id_num, source_data_dir, destination_data_dir, mode, model_path, split, arch, size_aware, epochs):
    """
    Train or fine-tune a SegmentClassifier on image data, then evaluate and save results.

    Optionally splits raw data into train/val/test sets before training. Supports
    training from scratch ("raw" mode) or fine-tuning from a pretrained checkpoint.
    After training, saves the model, plots the training history, and outputs a
    confusion matrix.

    Parameters
    ----------
    id_num : str
        A string prefix used to construct a unique run ID (combined with the
        current datetime).
    source_data_dir : str
        Path to the unsplit source data directory. Only used when split=True.
        Expected structure:
            source_data_dir/
                <label_1>/
                    file1, file2, ...
                <label_2>/
                    ...
    destination_data_dir : str
        Path to the directory containing (or to receive) the train/, val/, and
        test/ subdirectories used for training and evaluation.
    mode : str
        Training mode. Use "raw" to train a new model from scratch. Any other
        value triggers fine-tuning from the checkpoint at model_path.
    model_path : str
        Path to a saved PyTorch model (.pt) to load for fine-tuning.
        Ignored when mode="raw".
    split : bool
        If True, calls split_data() to partition source_data_dir into
        train/val/test splits before training.
    arch : str
        Backbone architecture identifier passed to classifier.load_model().
    size_aware : bool
        If True, computes per-channel mean and standard deviation of nuclei
        per bounding box (NPB) from the training set and passes them to
        SegmentClassifier for size-aware sampling. If False, MEAN_NPB and
        STD_NPB are set to None.

    Raises
    ------
    FileNotFoundError
        If destination_data_dir or (when mode != "raw") model_path does not exist.
    """
    Transform = transforms.Compose([
        transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        # transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if split:
        filenames, num_files = split_data(source_data_dir, destination_data_dir)

    if size_aware:
        MEAN_NPB, STD_NPB = calculate_mean_std_npb(os.path.join(destination_data_dir, "train"))
    else:
        MEAN_NPB, STD_NPB = None, None

    print(f"Creating SegmentClassifier")
    run_id = id_num + datetime.today().strftime("%m-%d-%H-%M")


    print(f"Loading model")
    if mode == "raw":
        classifier = SegmentClassifier(id=run_id, data_dir=destination_data_dir, num_classes=5,
                                       device=device, optim=2,
                                       lr=1e-2, batch_size=32, num_workers=4, Transform=Transform, sample=True,
                                       loss_weights=True, mean_npb=MEAN_NPB, std_npb=STD_NPB)
        print(f"Loading data")

        train_loader, val_loader = classifier.load_data()
        classifier.load_model(backbone=arch)
        print(f"Fitting SegmentClassifier")
        history = classifier.fit(num_epochs=epochs, unfreeze_after=10, train_loader=train_loader, val_loader=val_loader)
        val_targets_labels = []
        val_preds_labels = []
        idx2class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
        for (target, pred) in zip(classifier.val_targets, classifier.val_predictions):
            val_targets_labels.append(idx2class[target.max()])
            val_preds_labels.append(idx2class[pred.max()])

        try:
            torch.save(classifier.model, os.path.join("models", f"SegmentClassifier_{run_id}.pt"))
        except:
            print("Could not save")


        # UNCOMMENT LATER
        # plot_training_history(history, run_id=run_id)
        cm = confusion_matrix(val_targets_labels, val_preds_labels)
        ConfusionMatrixDisplay(cm, display_labels=list(val_loader.dataset.class_to_idx.keys())).plot()
        plt.savefig(os.path.join("outputs", "train", f"confusion_matrix_{run_id}.png"))

        # current_datetime = datetime.datetime.now()
        print("stop")
    else:
        classifier = SegmentClassifier(id=run_id, data_dir=destination_data_dir, num_classes=5,
                                       device=device, optim=2,
                                       lr=1e-2, batch_size=32, num_workers=4, Transform=Transform, sample=True,
                                       loss_weights=True, mean_npb=MEAN_NPB, std_npb=STD_NPB)
        pretrained = torch.load(model_path)
        train_loader, val_loader = classifier.load_data()
        classifier.load_model(pretrained, backbone=arch)
        print(f"Fitting pretrained SegmentClassifier")
        history = classifier.fit(num_epochs=1, unfreeze_after=10, train_loader=train_loader, val_loader=val_loader)
        print("Finished")
    # else:
    #     test_data_dir = "C:\\Users\\ALANalysis\\flat-bug\\src\\labeller\\data_test_set\\test"
    #     pretrained = torch.load("C:\\Users\\ALANalysis\\flat-bug\\src\\labeller\\SegmentClassifier_2025_03_27.pt", weights_only=False)
    #     pretrained.eval()
    #
    #     class Classifier_Test(Dataset):
    #         def __init__(self, dir, transform=None):
    #             self.dir = dir
    #             self.transform = transform
    #             self.images = os.listdir(self.dir)
    #
    #         def __len__(self):
    #             return len(self.images)
    #
    #         def __getitem__(self, index):
    #             # print(os.path.join(self.dir, self.images[idx]))
    #             img = Image.open(os.path.join(self.dir, self.images[index]))
    #             return self.transform(img), self.images[index]
    #
    #     # check whether val_set results match last epoch or best epoch
    #     val_set = Classifier_Test(test_data_dir, transform=transforms.Compose([
    #         transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    #         transforms.Resize(size=(224, 224), antialias=True),
    #         transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    #         transforms.Lambda(lambda x: x[:3])
    #         # remove alpha channel since the model's dataset is built with ImageFolder which is RGB
    #     ]))
    #
    #     val_loader = DataLoader(val_set, batch_size=32)
    #
    #     sub = pd.DataFrame(columns=['category', 'id'])
    #     id_list = []
    #     pred_list = []
    #
    #     pretrained = pretrained.to(device)
    #
    #     with torch.no_grad():
    #         for (image, image_id) in val_loader:
    #             image = image.to(device)
    #
    #             logits = pretrained(image)
    #             predicted = list(torch.argmax(logits, 1).cpu().numpy())
    #
    #             for id in image_id:
    #                 id_list.append(id)
    #
    #             for prediction in predicted:
    #                 pred_list.append(prediction.tolist())
    #
    #     sub['category'] = pred_list
    #     sub['id'] = id_list
    #
    #     mapping = {0: 'Non-target', 1: 'SWD_male', 2: 'SWD_parasitoid', 3: 'Weevil'}
    #
    #     sub['category'] = sub['category'].map(mapping)
    #     sub = sub.sort_values(by='id')
    #     sub.to_csv(test_data_dir + "train.csv", index=False)

def main():
    classify(**cli_args())

if __name__ == '__main__':
    main()
