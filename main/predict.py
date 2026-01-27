import torch
import pandas as pd
from dataset import XrayDataset
from multimodel import Multimodel
from transform import val_transform
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path= ''
    img_dir = ''
    test_csv = ''

    model = Multimodel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    test_dataset = XrayDataset(img_dir, test_csv, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    predictions = []
    true_labels = []
    image_names = []

    with torch.no_grad():
        for images, labels, _, _ in test_loader:
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
    df.to_csv('predictions.csv', index=False)

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