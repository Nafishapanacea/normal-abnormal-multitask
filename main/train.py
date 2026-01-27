import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import XrayDataset
from multimodel import Multimodel
from transform import train_transform, val_transform
from utils import train_one_epoch, validate

img_dir = ''
train_csv = r"C:\Users\Acer\Desktop\Office\X-ray-NormalVsAbnormal\Normal-abnormal-multimodel\test.csv"
val_csv = r"C:\Users\Acer\Desktop\Office\X-ray-NormalVsAbnormal\Normal-abnormal-multimodel\test.csv"

epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    train_dataset = XrayDataset(img_dir, train_csv, transform=train_transform)
    val_dataset = XrayDataset(img_dir, val_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    model = Multimodel().to(device)
    criterian = nn.BCEWithLogitsLoss()
    bbox_loss = nn.MSELoss()

    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    bbox_params = list(model.bbox_head.parameters())

    optimizer_cls = torch.optim.Adam(
        backbone_params + classifier_params,
        lr=1e-4
    )

    optimizer_bbox = torch.optim.Adam(
        bbox_params,
        lr=1e-4
    )
    least_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer_cls, optimizer_bbox, criterian, bbox_loss, device)
        val_loss, val_acc = validate(model, val_loader, criterian, device)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch {epoch+1}: New best model saved with val_loss: {val_loss:.4f} and val_acc: {val_acc:.4f}')
    
    torch.save(model.state_dict(), 'last_model.pth')
    print('Training complete. Last model saved as last_model.pth')

if __name__ == '__main__':
    train()

        

