import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.v2 as transforms
from torchvision.transforms import RandomRotation

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from label_crops import SegmentClassifier


def cli_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained checkpoint (.pt)"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Dataset directory containing train/ val/ test/"
    )

    parser.add_argument(
        "--arch",
        default="resnet50"
    )

    parser.add_argument(
        "--num_classes",
        required=True,
        type=int
    )

    return parser.parse_args()


def main():

    args = cli_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    classifier = SegmentClassifier(
        id="test",
        data_dir=args.data,
        num_classes=args.num_classes,
        device=device,
        batch_size=32,
        num_workers=4,
        Transform=transform,
        sample=False,
        loss_weights=False,
    )

    checkpoint = torch.load(args.model, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    train_loader, val_loader = classifier.load_data()

    classifier.load_model(pretrained=state_dict, backbone=args.arch)

    model = classifier.model
    model.eval()

    test_dataset = ImageFolder(
        os.path.join(args.data, "test"),
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())


    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}\n")

    report = classification_report(
        y_true,
        y_pred,
        target_names=test_dataset.classes,
        digits=4,
    )

    print(report)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=test_dataset.classes,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)

    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png")

    with open("classification_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    print("Saved:")
    print("  classification_report.txt")
    print("  test_confusion_matrix.png")


if __name__ == "__main__":
    main()