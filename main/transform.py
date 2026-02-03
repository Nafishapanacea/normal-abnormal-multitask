import torchvision.transforms as T
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A

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
])

val_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



train_transform_bbox = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format="pascal_voc",
        label_fields=["bbox_labels"],
        # clip=True
    )
)

train_transform_nobbox = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ]
)