import torch
import pandas as pd
from dataset import XrayDataset
from multimodel import Multimodel
from transform import val_transform
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def predict():
    img_dir = ''
    test_csv = ''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = ''

    model = Multimodel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location = device))
    model.eval()

    test_dataset = XrayDataset(img_dir, test_csv, transform= val_transform)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle=False, num_workers=4)

    true_labels = test_dataset.df['label'].map({'Normal':0, "Abnormal":1}).tolist()
    image_name = test_dataset.df['image_id'].tolist()

    predictions = []
    with torch.no_grad():
        for images, label, _ in test_loader:
            images = images.to(device)
            output = model(images)

            preds = (torch.sigmoid(output).squeeze(1) >0.5).float()
            predictions.extend(preds.cpu().numpy().tolist())

    df = pd.DataFrame({
        'image_name': image_name,
        'true_label': true_labels,
        'predicted_label': predictions
    })
   
    df.to_csv('predictions.csv', index = False)
    print('Predictions saved to predictions.csv')

    # Evaluation Metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)   
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)

    tn, fp, fn, tp = cm.ravel()
    specificity = tn/(tn + fp)

    print("\n==== Evaluation Metrics ====")
    print(f"Accuracy     :  {accuracy:.4f}")
    print(f"Precision    :  {precision:.4f}")
    print(f"Recall       :  {recall:.4f}")
    print(f"Specificity  :  {specificity:.4f}")
    print(f"F1 Score     :  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)


if __name__ == '__main__':
    predict()