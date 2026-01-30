import os
import torch
from PIL import Image
import pandas as pd
# from utils import encode_disease
from torch.utils.data import Dataset
from config import disease2id

class XrayDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform = None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.img_dir, image_id)
        
        # image = Image.open('/home/jupyter-nafisha/normal-abnormal-multitask/padchest_normalized.png').convert('RGB')
        image = Image.open(image_path).convert('RGB')

        label = row['label']
        disease_name = row['class_name']

        x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]

        if any(pd.isna([x_min, y_min, x_max, y_max])):
            bbox = torch.zeros(4, dtype=torch.float32)
            has_bbox = torch.tensor(0, dtype=torch.bool)
            disease_id= disease2id['no_bbox']
        else:
            bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            has_bbox = torch.tensor(1, dtype=torch.bool)
            disease_id = disease2id[disease_name]

        disease_id = torch.tensor(disease_id, dtype=torch.long)
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, disease_id, label, bbox, has_bbox