from PIL import Image
import numpy as np
import torchvision.transforms as T

train_transforms = T.Compose([

    # Rotation ±10–15 degrees
    # T.RandomRotation(degrees=15),

    # # Translation ±10–15%
    # T.RandomAffine(
    #     degrees=0,
    #     translate=(0.1, 0.15),
    #     scale=(0.9, 1.1) 
    # ),

    # Brightness / contrast jitter
    T.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

])