from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from dataset import XrayDataset
from multimodel import Multimodel
from utils import train_one_epoch, validate
from transform import train_transforms
from transformers import AutoModel, AutoProcessor, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)
vision_full = AutoModel.from_pretrained(
    MODEL_NAME,
    config=config,
    trust_remote_code=True
).to(device, dtype)
vision_encoder = vision_full.vision_model
del vision_full


# img_dir = '/home/jupyter-nafisha/Data/data_v3_CLAHE'
# train_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/train.csv'
# train_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/vinbig_balanced_100.csv'
# val_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/val_withoutBbox_subset.csv'

img_dir = ''
train_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/trainWithTB.csv'
val_csv= '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/valWithTB.csv'

epochs = 12

def train():
    train_dataset = XrayDataset(img_dir, train_csv, transform=train_transforms)
    val_dataset = XrayDataset(img_dir, val_csv, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = Multimodel(vision_encoder= vision_encoder).to(device)
    pos_weight = torch.tensor([0.75], device=device)
    criterian = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # criterian = nn.BCEWithLogitsLoss()
    bbox_loss = nn.MSELoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    least_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterian, bbox_loss, device)
        val_loss, val_acc = validate(model, val_loader, criterian, device)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}: New best model saved with val_loss: {val_loss:.4f} and val_acc: {val_acc:.4f}')

        # break
    
    torch.save(model.state_dict(), 'last_model.pth')
    print('Training complete. Last model saved as last_model.pth')

if __name__ == '__main__':
    train()