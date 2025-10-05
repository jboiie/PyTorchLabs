import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#feed forward neural network. with two hidden layer.

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
    
torch.manual_seed(41) #picking a manual seed for randomizaation

mod = model()
#print(model)
csv_path = r"C:\Programming\Projects\Learn\pytorch\archive\Iris.csv"
df = pd.read_csv(csv_path)

df['Species'] = df['Species'].replace('Iris-setosa', 0)
df['Species'] = df['Species'].replace('Iris-versicolor', 1)
df['Species'] = df['Species'].replace('Iris-virginica', 2)

#print(df.head)

# train test and split now
X=df.drop(['Id','Species'], axis=1)
y=df['Species']

X = X.values
y = y.values

#print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=41)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)#y and x labels to tensors.
y_test = torch.LongTensor(y_test)

#criteria to measure the error
criterion = nn.CrossEntropyLoss()
#adam optimizer
optimizer = torch.optim.Adam(mod.parameters(), lr=0.01)

#print(mod.parameters)

#train the model.
epochs=100
losses = []
for i in range(epochs):
    y_pred = mod.forward(X_train) # go forward and predict
    loss =criterion(y_pred, y_train) #predicted vs y_train
    losses.append(loss.detach().numpy())#keeping track 
    if i%10==0:
        print(f'Epoch:{i} and loss: {loss}') #print every ten epochs
    # back propogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
