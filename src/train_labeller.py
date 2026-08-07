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
    -n, --num_classes
        Number of classes to classify
    -e, --epochs
        Number of epochs to train over
    -t, --learning_rate
        Learning rate while training the model

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
    args_parse.add_argument("-r", "--arch", type=str, dest="arch", default="resnet50",
                            help="Model architecture (resnet, bioclip)")
    args_parse.add_argument("-a", "--size_aware", dest="size_aware", action='store_true',
                            help="Whether to perform size-aware classification")
    args_parse.add_argument("-n", "--num_classes", type=int, dest="num_classes", required=True,
                            help="Number of classes for this run")
    args_parse.add_argument("-e", "--epochs", type=int, dest="epochs", required=True,
                            help="Number of epochs")
    args_parse.add_argument("-t", "--learning_rate", type=float, dest="learning_rate", required=True,
                            help="Learning Rate")
    args = args_parse.parse_args()
    return vars(args)


def plot_training_history(history, run_id, outputs_train_dir):
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
    plt.savefig(os.path.join(outputs_train_dir, f"training_history_{run_id}.png"))

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

def classify(id_num, source_data_dir, destination_data_dir, mode, model_path, split, arch, size_aware, num_classes, epochs, learning_rate):
    """
    Train or fine-tune a SegmentClassifier on image data, then evaluate and save results.

    Optionally splits raw data into train/val/test sets before training. Supports
    training from scratch ("raw" mode) or fine-tuning from a pretrained checkpoint.
    After training, saves the model with NPB values embedded in the checkpoint,
    plots the training history, and outputs a confusion matrix.

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
    num_classes : int
        Number of output classes for classification.
    epochs : int
        Number of training epochs.
    learning_rate : float
        Learning rate for the optimizer.

    Raises
    ------
    FileNotFoundError
        If destination_data_dir or (when mode != "raw") model_path does not exist.
    """
    # Basic Augmentation
    # Transform = transforms.Compose([
    #     transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    #     # transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.RandomVerticalFlip(p=0.5),
    #     transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
    #     transforms.Resize(size=(224, 224), antialias=True),
    #     transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    Transform = transforms.Compose([
        transforms.ToImage(),
        
        # Geometric transformations
        transforms.RandomHorizontalFlip(p=0.6),      # Increase from 0.5
        transforms.RandomVerticalFlip(p=0.6),        # Increase from 0.5
        transforms.RandomApply([RandomRotation((90, 90))], p=0.6),
        
        # Color transformations
        transforms.ColorJitter(
            brightness=0.2,      # ±20% brightness
            contrast=0.2,        # ±20% contrast
            saturation=0.2,      # ±20% saturation
            hue=0.05             # ±5% hue
        ),
        
        # Resize and normalize
        transforms.Resize(size=(224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 4
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if split:
        filenames, num_files = split_data(source_data_dir, destination_data_dir)

    print(f"Creating SegmentClassifier")
    run_id = id_num + datetime.today().strftime("%m-%d-%H-%M")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_dir = os.path.join(models_dir, f"checkpoint_{run_id}.pt")
    print(f"Checkpoints will be saved to {checkpoint_dir}")

    outputs_train_dir = os.path.join(project_root, "outputs", "train")
    os.makedirs(outputs_train_dir, exist_ok=True)

    print(f"Loading model")
    if mode == "raw":
        classifier = SegmentClassifier(id=run_id, data_dir=destination_data_dir, num_classes=num_classes,
                                       device=device, optim=2,
                                       lr=learning_rate, batch_size=32, num_workers=4, Transform=Transform, sample=True,
                                       loss_weights=True)
        print(f"Loading data")

        train_loader, val_loader = classifier.load_data()
        classifier.load_model(backbone=arch)
        print(f"Fitting SegmentClassifier")
        history = classifier.fit(num_epochs=epochs, unfreeze_after=10, train_loader=train_loader, val_loader=val_loader, checkpoint_dir=checkpoint_dir)
        val_targets_labels = []
        val_preds_labels = []
        idx2class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
        for (target, pred) in zip(classifier.val_targets, classifier.val_predictions):
            val_targets_labels.append(idx2class[target.max()])
            val_preds_labels.append(idx2class[pred.max()])

        # Save checkpoint with NPB values embedded
        enhanced_checkpoint = {
            'state_dict': classifier.model.state_dict(),
            'mean_npb': classifier.mean_npb,
            'std_npb': classifier.std_npb,
            'architecture': arch,
            'num_classes': num_classes,
            'run_id': run_id,
            'size_aware': size_aware,
            'epoch': epochs,
        }
        
        model_save_path = os.path.join(models_dir, f"SegmentClassifier_{run_id}.pt")
        try:
            torch.save(enhanced_checkpoint, model_save_path)
            print(f"  Saved model checkpoint with NPB values:")
            print(f"  Path: {model_save_path}")
            print(f"  Mean NPB: {classifier.mean_npb:.2f}")
            print(f"  Std NPB: {classifier.std_npb:.2f}")
            print(f"  Architecture: {arch}")
        except Exception as e:
            print(f" Error saving checkpoint: {e}")

        # UNCOMMENT LATER
        # plot_training_history(history, run_id=run_id, outputs_train_dir=outputs_train_dir)
        cm = confusion_matrix(val_targets_labels, val_preds_labels)
        cmp = ConfusionMatrixDisplay(cm, display_labels=list(val_loader.dataset.class_to_idx.keys()))
        fig, ax = plt.subplots(figsize=(15,15))
        cmp.plot(ax=ax)
        plt.savefig(os.path.join(outputs_train_dir, f"confusion_matrix_{run_id}.png"))

        # current_datetime = datetime.datetime.now()
        print("Training complete!")
    else:
        classifier = SegmentClassifier(id=run_id, data_dir=destination_data_dir, num_classes=num_classes,
                                       device=device, optim=2,
                                       lr=learning_rate, batch_size=32, num_workers=4, Transform=Transform, sample=True,
                                       loss_weights=True)
        
        # Load checkpoint - handle both new enhanced format and old state_dict format
        checkpoint_data = torch.load(model_path)
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
            # Enhanced checkpoint format with NPB values
            pretrained = checkpoint_data['state_dict']
            if 'mean_npb' in checkpoint_data and 'std_npb' in checkpoint_data:
                classifier.mean_npb = checkpoint_data['mean_npb']
                classifier.std_npb = checkpoint_data['std_npb']
                print(f" Loaded pretrained model with NPB values:")
                print(f"  Mean NPB: {classifier.mean_npb:.2f}")
                print(f"  Std NPB: {classifier.std_npb:.2f}")
        else:
            # Legacy state_dict format - mean_npb/std_npb will be calculated from training data
            pretrained = checkpoint_data
            print(" Loaded legacy checkpoint (no NPB values stored)")
            print(f"  NPB values will be recalculated from training data")
        
        train_loader, val_loader = classifier.load_data()
        classifier.load_model(pretrained, backbone=arch)
        print(f"Fitting pretrained SegmentClassifier")
        history = classifier.fit(num_epochs=epochs, unfreeze_after=10, train_loader=train_loader, val_loader=val_loader, checkpoint_dir=checkpoint_dir)
        
        # Save finetuned checkpoint with NPB values
        enhanced_checkpoint = {
            'state_dict': classifier.model.state_dict(),
            'mean_npb': classifier.mean_npb,
            'std_npb': classifier.std_npb,
            'architecture': arch,
            'num_classes': num_classes,
            'run_id': run_id,
            'size_aware': size_aware,
            'epoch': epochs,
        }
        
        model_save_path = os.path.join(models_dir, f"SegmentClassifier_{run_id}_finetuned.pt")
        try:
            torch.save(enhanced_checkpoint, model_save_path)
            print(f" Saved finetuned checkpoint with NPB values:")
            print(f" Path: {model_save_path}")
        except Exception as e:
            print(f" Error saving finetuned checkpoint: {e}")
        
        print("Finetuning complete!")

def main():
    classify(**cli_args())

if __name__ == '__main__':
    main()