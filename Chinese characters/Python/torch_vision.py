
import torch
import torchvision.models as models

import characters_learning as ch_l
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# %% Data loading
data_folder_path = "D:/Machine Learning/Chinese project/handwritten chinese numbers"
current_folder_path = os.path.dirname(os.path.abspath(__file__))

data_csv = pd.read_csv(data_folder_path+"/chinese_mnist.csv")
DataFrame_csv = pd.DataFrame(data_csv)
keys = DataFrame_csv.columns

suite_id = DataFrame_csv[keys[0]].values
sample_id = DataFrame_csv[keys[1]].values
code = DataFrame_csv[keys[2]].values
value = DataFrame_csv[keys[3]].values
value[value == 100 ] = 11
value[value == 1000] = 12
value[value == 10000] = 13
value[value == 100e6] = 14
print(np.unique(value))

# %%
# Define the Spatial Transformer Network (STN) module
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        
        return x

# Define the model combining STN and classification
class STNClassifier(nn.Module):
    def __init__(self):
        super(STNClassifier, self).__init__()
        self.stn = STN()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.stn(x)
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# %%

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("plain_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument

    tensor = ch_l.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, m, n) = ch_l.TensorToModelTensors()

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("model_training" in sys.argv) ):     

    # Initialize and train the STNClassifier
    model = STNClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = int(input("Number of epochs : "))
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = F.nll_loss(output, train_labels)
        loss.backward()
        optimizer.step()

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT TERMINATED SUCCESSFULLY\033[0m")