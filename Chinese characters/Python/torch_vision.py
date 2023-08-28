
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
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self, batch_size, in_channels=1):
        super(STN, self).__init__()
        self.batch_size = batch_size

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Localization head (predicts the affine transformation)
        self.fc_loc = nn.Sequential(
            nn.Linear(int(self.batch_size*24/125), 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights and biases for the localization network
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, int(self.batch_size*24/125)) #1440
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        
        return x

class STNClassifier(nn.Module):
    def __init__(self, batch_size, in_channels=1, height=64, width=64, num_classes=15):
        super(STNClassifier, self).__init__()
        
        self.stn = STN(batch_size=batch_size, in_channels=in_channels)
        
        self.fc1 = nn.Linear(height * width, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.stn(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# %%

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("raw_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    tensor = ch_l.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convoluted_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    tensor = ch_l.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("contrasted_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    tensor = ch_l.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("rotated_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    tensor = ch_l.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()

# %%

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("model_training" in sys.argv) ):     

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()
    batch_size = z
    num_classes = len(np.unique(train_labels))
    
    model = STNClassifier(batch_size=batch_size, in_channels=in_channels, height=m, width=n, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #100epochs lr=0.0001 89%

    running_loss = 0
    epochs = int(input("Number of epochs : "))
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = F.nll_loss(output, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# %%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

class STNModel(nn.Module):
    def __init__(self, batch_size, in_channels):
        super(STNModel, self).__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.resnet18 = models.resnet18()
        
        self.localization = nn.Sequential(
            nn.Conv2d(self.in_channels, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(int(self.batch_size*24/125), 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 3)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        xs = self.localization(x)
        xs = xs.view(-1, int(self.batch_size*24/125))
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


# %%
if __name__ == "__main__" and ("builtin" in sys.argv):

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = ch_l.TensorToTensors()
    batch_size = z
    num_classes = len(np.unique(train_labels))
    
    model = STNModel(batch_size=batch_size, in_channels=in_channels)

    input_image = train_data
    input_image = torch.randn(7500, 3, m, n)


    stn_transform = transforms.Compose([
        transforms.ToPILImage(mode="L"),
        transforms.Resize((64, 64)), 
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Lambda(lambda x: x.unsqueeze(0))  # Add batch dimension
    ])

    theta = model(input_image)
    output_image = transforms.functional.grid_sample(input_image, theta, align_corners=False)
    #output_image = stn_transform(output_image[0])  # Remove batch dimension

    # Display the input and output images
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(transforms.ToPILImage()(input_image[0]))
    plt.title("Input Image")
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title("STN Transformed Image")
    plt.show()

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT TERMINATED SUCCESSFULLY\033[0m")
