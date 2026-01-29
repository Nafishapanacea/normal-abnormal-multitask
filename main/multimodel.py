import torch
import torch.nn as nn
from torchvision import models
from config import NUM_DISEASE_TYPE

class Multimodel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = models.densenet121(pretrained = True)
        num_feats = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.disease_embeddings = nn.Embedding(NUM_DISEASE_TYPE, 128)

        self.classifier = nn.Sequential(
            nn.Linear(num_feats, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(num_feats+128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,4)
        )

    def forward(self, img, disease_id= None, return_bbox=False):
        feats = self.backbone(img)
        cls_logits = self.classifier(feats)

        if not return_bbox:
            return cls_logits

        disease_embeddings = self.disease_embeddings(disease_id)
        combined = torch.cat([feats, disease_embeddings], dim=1)
        
        bbox_preds = self.bbox_head(combined)

        # if has_bbox is not None:
        #     bbox_preds = bbox_preds * has_bbox.unsqueeze(1).float()

        return cls_logits, bbox_preds

