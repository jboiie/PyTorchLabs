import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class StudentCSVLoader(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.feature_cols = ["Age", "StudyHours", "Marks"] #selective columns
        self.data["Label"] = self.data["Passed"].map({"Yes": 1, "No": 0}) # passed columns into numeric labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = self.data.loc[idx, self.feature_cols].values.astype('float32')
        label = self.data.loc[idx, "Label"]
        return torch.tensor(features), torch.tensor(label)

csv_path = r'C:\Programming\Projects\Learn\pytorch\sample_data.csv'
dataset = StudentCSVLoader(csv_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_idx, (features, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}")
    print("Features:", features)
    print("Labels:", labels)
