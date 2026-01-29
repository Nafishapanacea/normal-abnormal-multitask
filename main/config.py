# epochs = 100
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# learning_rate = 

# img_dir = ''
# train_csv = r"C:\Users\Acer\Desktop\Office\X-ray-NormalVsAbnormal\Normal-abnormal-multimodel\test.csv"
# val_csv = r"C:\Users\Acer\Desktop\Office\X-ray-NormalVsAbnormal\Normal-abnormal-multimodel\test.csv"
# test_csv = ''
# checkpoint_path= ''
# prediction_csv = ''

NUM_DISEASE_TYPE = 16

disease2id = {
    'no_bbox': 0,
    'No finding': 1,
    'Aortic enlargement': 2,
    'Atelectasis': 3,
    'Calcification': 4,
    'Cardiomegaly': 5,
    'Consolidation': 6,
    'ILD': 7,
    'Infiltration': 8,
    'Lung Opacity': 9,
    'Nodule/Mass': 10,
    'Other lesion': 11,
    'Pleural effusion': 12,
    'Pleural thickening': 13,
    'Pneumothorax': 14,
    'Pulmonary fibrosis': 15
}