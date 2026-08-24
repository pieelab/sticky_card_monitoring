import torch, os, sys, glob, shutil
from os.path import join
os.chdir('C:/Users/ALANalysis/flat-bug/src/labeller')
sys.path.insert(0, os.getcwd())
from PIL import Image
from label_crops import SegmentClassifier
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms              # composable transforms
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "./dataset_4class_small_unclassified"
destination_data_dir = "./dataset_4class_small_classified_by_labeller"
pretrained = torch.load("./SegmentClassifier_2025_03_27.pt", weights_only = False)
pretrained.eval()


class Classifier_Test(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.transform = transform
        self.images = os.listdir(self.dir)

    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        #print(os.path.join(self.dir, self.images[idx]))
        img = Image.open(join(self.dir, self.images[index]))
        return self.transform(img), self.images[index]
    
    
# check whether val_set results match last epoch or best epoch
val_set = Classifier_Test(data_dir, transform = transforms.Compose([
            transforms.ToImage(),  # Convert to tensor, only needed if you had a PIL image
            transforms.Resize(size = (224, 224), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            transforms.Lambda(lambda x: x[:3]) # remove alpha channel since the model's dataset is built with ImageFolder which is RGB
            ]))

#val_loader = DataLoader(val_set, batch_size = len(val_set))
val_loader = DataLoader(val_set, batch_size = 16)

sub = pd.DataFrame(columns = ['category', 'id'])
id_list = []
pred_list = []

mapping = {0:'Other', 1:'SWD_male', 2:'SWD_parasitoid', 3:'Weevil'}
#mapping = {0: 'Arthropod', 1: 'Debris', 2: 'SWD_male', 3: 'SWD_parasitoid', 4: 'Small_black_weevil'}

# make subfolders for all classes
for key, subfolder in mapping.items():
    sub_path = join(destination_data_dir, subfolder)
    if not os.path.isdir(sub_path):
        os.mkdir(join(sub_path))

pretrained = pretrained.to(device)

with torch.no_grad():
    for (image, image_id) in val_loader:
        image = image.to(device)

        logits = pretrained(image)
        predicted = list(torch.argmax(logits, 1).cpu().numpy())

        # for id in image_id:
        #     id_list.append(id)
        #
        # for prediction in predicted:
        #     pred_list.append(prediction.tolist())
        for id, prediction in zip(image_id, predicted):
            id_list.append(id)
            pred_list.append(prediction.tolist())
            shutil.copy(join(data_dir, id),
                        join(destination_data_dir, mapping[prediction]))


sub['category'] = pred_list
sub['id'] = id_list


sub['category'] = sub['category'].map(mapping)
sub = sub.sort_values(by='id')

