from v2025_label_crops import SegmentClassifier, ImageFolderWithPaths
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sn
import pandas as pd
import torch
import torchvision.transforms.v2  as transforms              # composable transforms
from torchvision.transforms import RandomRotation
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == '__main__':
    Transform = transforms.Compose([
        transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
        #transforms.ToDtype(torch.uint8, scale=True),  # optional, most input are already uint8 at this point
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([RandomRotation((90, 90))], p=0.5),
        transforms.Resize(size = (224, 224), antialias=True),
        transforms.ToDtype(torch.float32, scale=True)  # Normalize expects float input
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = "./data"
    run_id = "5.0_" + datetime.today().strftime("%m-%d-%H-%M")
    classifier = SegmentClassifier(id = run_id, data_dir = data_dir, num_classes = 5,
                                   device = device, optim = 1,
                                   lr = 1e-4, batch_size = 16, num_workers = 4, Transform = Transform, sample = True,
                                   loss_weights = True)
    train_loader, val_loader = classifier.load_data()
    classifier.load_model()
    classifier.fit(num_epochs = 100, unfreeze_after = 5, train_loader = train_loader, val_loader = val_loader)
    val_targets_labels = []
    val_preds_labels = []
    idx2class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}

    for (target, pred) in zip(classifier.val_targets, classifier.val_predictions):
        val_targets_labels.append(idx2class[target.max()])
        val_preds_labels.append(idx2class[pred.max()])

    cm = confusion_matrix(val_targets_labels, val_preds_labels)
    ConfusionMatrixDisplay(cm, display_labels = list(val_loader.dataset.class_to_idx.keys())).plot()
    plt.show()
    try:
        torch.save(classifier.model, f"SegmentClassifier_{run_id}.pt")
    except:
        print("Could not save") 
    print("stop")

#    classifier.compute_confusion_matrix()




# try:


#     torch.save(classifier.model, "SegmentClassifier.pt")
# except:
#     print("Could not save") 
