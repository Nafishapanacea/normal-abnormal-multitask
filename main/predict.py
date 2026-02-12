import torch
import pandas as pd
from dataset import XrayDataset
from multimodel import Multimodel
# from transform import val_transform
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

def predict():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path= '/home/jupyter-nafisha/normal-abnormal-multitask/main/best_model.pth'
    
    img_dir = '/home/jupyter-nafisha/Data/data_v3_CLAHE'
    test_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/test_withoutBbox.csv'

    # padchest
    # img_dir = '/home/jupyter-nafisha/X-ray-covariates/padchest_normalized'
    # test_csv = '/home/jupyter-nafisha/normal-abnormal-multitask/CSVs/padchest_withoutBbox.csv'

    model = Multimodel(vision_encoder = vision_encoder).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dataset = XrayDataset(img_dir, test_csv, transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )

    predictions = []
    true_labels = []
    image_names = []

    with torch.no_grad():
        for images, _, labels, _, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.int().tolist())

    image_names = test_dataset.df['image_id'].tolist()

    df = pd.DataFrame({
        'image_name': image_names,
        'true_label': true_labels,
        'predicted_label': predictions
    })
    df.to_csv('predictions_originalTestSet_chexagent.csv', index=False)

    # Metrics
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy    : {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision   : {precision_score(true_labels, predictions):.4f}")
    print(f"Recall      : {recall_score(true_labels, predictions):.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1 Score    : {f1_score(true_labels, predictions):.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == '__main__':
    predict()