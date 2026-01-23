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

    def forward(self, img, has_bbox=None):
    
        feats = self.backbone(img)           # [B, num_feats]

        cls_logits = self.classifier(feats)  # [B,1]
        bbox_preds = self.bbox_head(feats)   # [B,4]

        # ---- Mask bbox output if no bbox exists ----
        # if has_bbox is not None:
        has_bbox = has_bbox.unsqueeze(1).float()  # [B,1]
        bbox_preds = bbox_preds * has_bbox        # zero out rows

        return cls_logits, bbox_preds

