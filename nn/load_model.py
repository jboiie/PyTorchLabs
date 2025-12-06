import torch
import torch.nn as nn
import torch.nn.functional as F
class model(nn.Module):
    #input layer(4features of the flower)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))#rectified linear unit
        x = F.relu(self.fc2(x))   
        x = self.out(x)

        return x
    

new_mod= model()
new_mod.load_state_dict(torch.load('iris_model.pt'))