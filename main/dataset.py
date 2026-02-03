import os
import cv2
import torch
from PIL import Image
import pandas as pd
import numpy as np
# from utils import encode_disease
from torch.utils.data import Dataset
from config import disease2id
from utils import has_valid_bbox

class XrayDataset(Dataset):
    def __init__(self, img_dir, csv_path, transform_bbox = None, transform_nobbox = None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform_bbox = transform_bbox
        self.transform_nobbox = transform_nobbox

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        image_path = os.path.join(self.img_dir, image_id)
        
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(r"C:\Users\Acer\Desktop\padchest_normalized.png").convert('RGB')
        image = np.array(image)

        label = row['label']
        disease_name = row['class_name']

        x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]

        bbox = [x_min, y_min, x_max, y_max]
        
        if has_valid_bbox(bbox) and self.transform_bbox:
            augmented = self.transform_bbox(
                image=image,
                bboxes=[bbox],
                bbox_labels=[0]   # dummy label (required by Albumentations)
            )
        
            image = augmented["image"]
            bbox = augmented["bboxes"][0]
            bbox = torch.tensor(bbox, dtype=torch.float32)
            has_bbox = torch.tensor(1, dtype=torch.bool)
            disease_id = disease2id[disease_name]
        
        else:
            augmented = self.transform_nobbox(image=image)
        
            image = augmented["image"]
            bbox = torch.zeros(4, dtype=torch.float32)
            has_bbox = torch.tensor(0, dtype=torch.bool)
            disease_id= disease2id['no_bbox']

        
        # if any(pd.isna([x_min, y_min, x_max, y_max])):
            # bbox = torch.zeros(4, dtype=torch.float32)
            # has_bbox = torch.tensor(0, dtype=torch.bool)
            # disease_id= disease2id['no_bbox']
        # else:
            # bbox = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)
            # has_bbox = torch.tensor(1, dtype=torch.bool)
            # disease_id = disease2id[disease_name]

        
        disease_id = torch.tensor(disease_id, dtype=torch.long)
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)

        return image, disease_id, label, bbox, has_bbox