import torch
import torch.nn as nn
from torchvision import models

class Multimodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.densenet121(pretrained = True)
        num_feats = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(num_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(num_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,4)
        )

    def forward(self, img, has_bbox=None, return_bbox=False):
        feats = self.backbone(img)
        cls_logits = self.classifier(feats)

        if not return_bbox:
            return cls_logits

        bbox_preds = self.bbox_head(feats)

        if has_bbox is not None:
            bbox_preds = bbox_preds * has_bbox.unsqueeze(1).float()

        return cls_logits, bbox_preds

