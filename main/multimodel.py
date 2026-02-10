import torch
import torch.nn as nn
from torchvision import models
from config import NUM_DISEASE_TYPE

class Multimodel(nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()

        self.vision_encoder = vision_encoder
        in_dim = vision_encoder.config.hidden_size

        self.disease_embeddings = nn.Embedding(NUM_DISEASE_TYPE, 128)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,1)
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(in_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512,4)
        )

    def forward(self, img, disease_id= None, return_bbox=False):
        outputs = self.vision_encoder(pixel_values = img, output_hidden_states=False, output_attentions=False)
        embeddings = outputs.pooler_output 
        
        cls_logits = self.classifier(embeddings)

        if not return_bbox:
            return cls_logits

        disease_embeddings = self.disease_embeddings(disease_id)
        combined = torch.cat([embeddings, disease_embeddings], dim=1)
        
        bbox_preds = self.bbox_head(combined)

        return cls_logits, bbox_preds

