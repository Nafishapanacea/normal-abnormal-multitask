from PIL import Image
import numpy as np
import torchvision.transforms as T
import albumentations as A

# train_transforms = T.Compose([

#     # Rotation ±10–15 degrees
#     # T.RandomRotation(degrees=15),

#     # # Translation ±10–15%
#     # T.RandomAffine(
#     #     degrees=0,
#     #     translate=(0.1, 0.15),
#     #     scale=(0.9, 1.1) 
#     # ),

#     # Brightness / contrast jitter
#     T.ColorJitter(
#         brightness=0.15,
#         contrast=0.15
#     ),

# ])

train_transforms = A.Compose([
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.ShiftScaleRotate(
        shift_limit=0.02,
        scale_limit=0.05,
        rotate_limit=7,
        border_mode=0,
        p=0.5   # force transform so you can see effect
    ),
],
bbox_params=A.BboxParams(
    format='albumentations',
    label_fields=['class_labels'],
    min_visibility=0.3
))