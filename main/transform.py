import torchvision.transforms as T
from PIL import Image
import numpy as np

train_transform = T.Compose([
    T.Resize((224,224)),
    T.RandomRotation(degrees=15),
    T.RandomAffine(
        degrees=0,
        translate=(0.1, 0.15),
        scale=(0.9,1.1)
    ),
    T.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])