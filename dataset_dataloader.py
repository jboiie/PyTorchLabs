import torch
from torch.utils.data import Dataset, DataLoader


# defining a custom dataset by subclassing torch.utils.data.Dataset
class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(10) #sample data
        self.labels = torch.arange(10)*2#sample labels, each element times 2
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    #for each index reutnr on data sample and its label
    
dataset = SimpleDataset() #instance of custom dataset

#gives 4 sample per batch, randomizes order each epoch(shuffle)
#dataloader for dataset
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# this loops over the dataloader, each iteration gives a batch of data and labels
for batch_idx, (data,labels) in enumerate(dataloader):
    print(f"batch {batch_idx}:")
    print("data: ", data)
    print("labels: ", labels)