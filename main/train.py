
import torch
import optuna
from torch import nn
from torch.utils.data import DataLoader
from dataset import XrayDataset
from multimodel import Multimodel
from utils import train_one_epoch, validate
from transform import train_transforms
from transformers import AutoModel, AutoConfig

MODEL_NAME = "StanfordAIMI/XraySigLIP__vit-l-16-siglip-384__webli"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

img_dir = ''
train_csv = '/home/ubuntu/Documents/Nafisha/chest-xray-NormalAbnormal/normal-abnormal-multitask/CSVs/trainWithTB-withAdditionalNormal.csv'
val_csv= '/home/ubuntu/Documents/Nafisha/chest-xray-NormalAbnormal/normal-abnormal-multitask/CSVs/valWithTB.csv'

EPOCHS = 8

    
# Dataset
train_dataset = XrayDataset(img_dir, train_csv, transform=train_transforms)
val_dataset = XrayDataset(img_dir, val_csv, transform=None)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

def objective(trial):

    # Parameters that can be tuned
    lr = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    pos_weight_val = trial.suggest_float("pos_weight", 1.0, 4.0)
    bbox_weight = trial.suggest_float("bbox_weight", 0.1, 1.0)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)

    # Reload the model after each trial
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

    # Model
    model = Multimodel(vision_encoder=vision_encoder).to(device)

    # change dropout dynamically
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout

    # Loss
    pos_weight = torch.tensor([pos_weight_val]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    bbox_loss = nn.MSELoss(reduction="none")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )


    best_val_loss = float("inf")

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            bbox_loss,
            bbox_weight,
            device
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(
            f"Trial {trial.number} Epoch {epoch+1} | "
            f"TrainLoss {train_loss:.4f} ValLoss {val_loss:.4f}"
        )

        break

    return best_val_loss


# ---------- Run Optuna ----------
if __name__ == "__main__":

    study = optuna.create_study(direction="minimize")

    study.optimize(
        objective,
        n_trials=20   # number of experiments
    )

    print("\nBest trial:")
    print(study.best_trial.params)
