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
from transformers import AutoProcessor

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

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
        
        image = Image.open(image_path).convert('RGB')
        orig_w, orig_h = image.size

        # Convert PIL → NumPy (Albumentations requirement)
        image_np = np.array(image)

        label = row['label']
        disease_name = row['class_name']

        x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]
        bbox = [x_min, y_min, x_max, y_max]
        print('image_path', image_path, 'and ', bbox)

        if has_valid_bbox(bbox):
            bbox = [
                x_min / orig_w,
                y_min / orig_h,
                x_max / orig_w,
                y_max / orig_h
            ]
            class_labels = [1]   # dummy label required by Albumentations
        else:
            bbox = []
            class_labels = []

        # --- Apply augmentation BEFORE processor ---
        if self.transform:
            transformed = self.transform(
                image=image_np,
                bboxes=[bbox] if bbox else [],
                class_labels=class_labels
            )

            image_np = transformed['image']

            if transformed['bboxes']:
                bbox = transformed['bboxes'][0]
                has_bbox = torch.tensor(1, dtype=torch.bool)
                disease_id = disease2id[disease_name]
            else:
                bbox = [0, 0, 0, 0]
                has_bbox = torch.tensor(0, dtype=torch.bool)
                disease_id = disease2id['no_bbox']
        else:
            if bbox:
                has_bbox = torch.tensor(1, dtype=torch.bool)
                disease_id = disease2id[disease_name]
            else:
                bbox = [0, 0, 0, 0]
                has_bbox = torch.tensor(0, dtype=torch.bool)
                disease_id = disease2id['no_bbox']
            
        inputs = processor(
            images=image_np,
            return_tensors="pt"
        )
        pixel_values = inputs['pixel_values'].squeeze(0)  
        bbox = torch.tensor(bbox, dtype=torch.float32)
        
        disease_id = torch.tensor(disease_id, dtype=torch.long)
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)

        return pixel_values, disease_id, label, bbox, has_bbox