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
        # image = Image.open(r"C:\Users\Acer\Desktop\Office\X-ray-NormalVsAbnormal\3FebDemoData\Abnormal\F_AP_Lung Opacity, Pleural Effusion.jpg").convert('RGB')

        # if self.transform:
        #     image = self.transform(image)
        orig_w, orig_h = image.size

        label = row['label']
        disease_name = row['class_name']

        x_min, y_min, x_max, y_max = row[["x_min", "y_min", "x_max", "y_max"]]
        bbox = [x_min, y_min, x_max, y_max]

        inputs = processor(
            images=image,
            return_tensors="pt"
        )
        pixel_values = inputs['pixel_values'].squeeze(0)  
        proc_h, proc_w = pixel_values.shape[-2:]

        if has_valid_bbox(bbox):
            scale_x = proc_w / orig_w
            scale_y = proc_h / orig_h

            x_min, y_min, x_max, y_max = bbox

            bbox = torch.tensor([
                x_min * scale_x,
                y_min * scale_y,
                x_max * scale_x,
                y_max * scale_y
            ], dtype=torch.float32)
            
            has_bbox = torch.tensor(1, dtype=torch.bool)
            disease_id = disease2id[disease_name]

        else:
            bbox = torch.zeros(4, dtype=torch.float32)
            has_bbox = torch.tensor(0, dtype=torch.bool)
            disease_id= disease2id['no_bbox']
        
        disease_id = torch.tensor(disease_id, dtype=torch.long)
        label = 0 if row['label'] == 'Normal' else 1
        label = torch.tensor(label, dtype=torch.float32)

        return pixel_values, disease_id, label, bbox, has_bbox