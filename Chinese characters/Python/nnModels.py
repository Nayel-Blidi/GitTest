# %% Imports / Pathing / sys argv / Data loading
import torch
import torchvision.models as models

import datasetGenerator as dg
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch.optim as optim

data_folder_path = "D:/Machine Learning/Chinese project/handwritten chinese numbers"
current_folder_path = os.path.dirname(os.path.abspath(__file__))

tensor_list = ["raw_tensor", "contrasted_tensor", "convoluted_tensor", "rotated_tensor"]
model_list = ["STNClassifier_model_training", "STNClassifier_model_testing", "resnet_training", "resnet_testing", "test"]
argv_list = tensor_list + model_list
for arg in sys.argv[1:]:
    if arg not in argv_list:
        raise ValueError(f"Wrong arg, try none or one of these :\n{tensor_list}\n{model_list}")

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("raw_tensor" in sys.argv) ):     

    data_tensor, targets = dg.DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    
    print(f"dtype: {data_tensor.dtype}")
    print(f"size: {data_tensor.size()}")
    print(f"mean: {torch.mean(data_tensor)}")
    print(f"max: {torch.max(data_tensor)}")


if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("contrasted_tensor" in sys.argv) ):     

    data_tensor, targets = dg.DatasetToContrastedTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    
    print(f"dtype: {data_tensor.dtype}")
    print(f"size: {data_tensor.size()}")
    print(f"mean: {torch.mean(data_tensor)}")
    print(f"max: {torch.max(data_tensor)}")    
    
if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convoluted_tensor" in sys.argv) ): 
        
    data_tensor, targets = dg.DatasetToConvolutedTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    
    print(f"dtype: {data_tensor.dtype}")
    print(f"size: {data_tensor.size()}")
    print(f"mean: {torch.mean(data_tensor)}")
    print(f"max: {torch.max(data_tensor)}")

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("rotated_tensor" in sys.argv) ):
         
    data_tensor, targets = dg.DatasetToRotatedTensor()
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    
    print(f"dtype: {data_tensor.dtype}")
    print(f"size: {data_tensor.size()}")
    print(f"mean: {torch.mean(data_tensor)}")
    print(f"max: {torch.max(data_tensor)}")
    

# %% Vanilla STN model + training/testing
if __name__ == "__main__" and ( ("STNClassifier_model_training" in sys.argv) or ("STNClassifier_model_testing" in sys.argv) ): # STNClassifier model class 

    print("class loading")
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

if __name__ == "__main__" and ("STNClassifier_model_training" in sys.argv): # STNClasifier training

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    batch_size = z
    num_classes = len(np.unique(train_labels))
    
    STNClassifier_model = STNClassifier(batch_size=batch_size, in_channels=in_channels, height=m, width=n, num_classes=num_classes)
    optimizer = torch.optim.Adam(STNClassifier_model.parameters(), lr=0.001) 
    #100epochs lr=0.0001 89% lr=0.01 78% lr=0.001 90% 
    #200epochs lr=0.001 92%
    
    print(STNClassifier_model)
    total_params = sum(p.numel() for p in STNClassifier_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:_}")
    
    running_loss = 0
    epochs = int(input("Number of epochs : "))
    for epoch in tqdm(range(epochs)):
        STNClassifier_model.train()
        optimizer.zero_grad()
        output = STNClassifier_model(train_data)
        loss = F.nll_loss(output, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

    torch.save(STNClassifier_model.state_dict(), f"STNClassifier_model_{epochs}.pth")
    print("Finished Training, model saved")

if __name__ == "__main__" and ("STNClassifier_model_testing" in sys.argv): # STNClasifier testing
    
    if ("STNClassifier_model_training" not in sys.argv):
        epochs = int(input("Model's number of epochs to load (choose among [200, 500]): "))

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    num_classes = len(np.unique(train_labels))
    
    STNClassifier_model = STNClassifier(batch_size=z, in_channels=in_channels, height=m, width=n, num_classes=num_classes)
    STNClassifier_model.load_state_dict(torch.load(f"STNClassifier_model_{epochs}.pth"))
    
    STNClassifier_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = STNClassifier_model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

# %% Resnet model + training/testing

if __name__ == "__main__" and ( [cmd_input for cmd_input in ["test1", "test2"] if cmd_input in sys.argv]):
    print("string in str.argv")


if __name__ == "__main__" and ( [cmd_input for cmd_input in ["resnet_training", "resnet_testing"] if cmd_input in sys.argv]): # Resnet model class
    
    class STNresnet18Model(nn.Module):
        def __init__(self, batch_size, in_channels):
            super(STNresnet18Model, self).__init__()
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

if __name__ == "__main__" and ( "data_loader" in sys.argv ) and ( ("resnet_training" in sys.argv) or ("resnet_testing" in sys.argv) ): # Resnet dataloader
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = datasets.ImageFolder(root="D:\Machine Learning\Chinese project\handwritten chinese numbers\dataloader_data", 
                                         transform=transform)
    
    dataset_length = len(dataset)
    print(dataset_length)
    train_dataset_length = round(dataset_length * 0.5)
    test_dataset_length = dataset_length - train_dataset_length
    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_dataset_length, test_dataset_length])
    
    dataloader = DataLoader(train_dataset, batch_size=75*5, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

if __name__ == "__main__" and ("resnet_training" in sys.argv): # Resnet training

    print("Step 2")
    resnet_model = models.resnet18(weights=None)
    num_classes = 15
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

    print("Step 3")
    lr, momentum = 0.01, 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet_model.parameters(), lr=lr, momentum=0.9)

    print("Step 4")
    num_epochs = int(input("Number of epochs : "))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet_model.to(device)

    print(f"Resnet model settings : num_classes={num_classes}")
    print(f"Resnet model settings : learning rate={lr}")
    print(f"Resnet model settings : momentum={momentum}")
    print(f"Resnet model settings : num_epochs={num_epochs}")
    resnet_model.print()

    batch_data, batch_labels = next(iter(dataloader))
    print(batch_data.size(), batch_labels.size())

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0  # Initialize loss for the epoch
        
        for i, (inputs, labels) in tqdm(enumerate(dataloader)):
            # inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = resnet_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss:.4f}")

    torch.save(resnet_model.state_dict(), 'resnet_model.pth')
    print("Finished Training, model saved")

if __name__ == "__main__" and ( "resnet_testing" in sys.argv): # Resnet testing

    num_classes=15
    resnet_model = models.resnet18()
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
    resnet_model.load_state_dict(torch.load('resnet_model.pth'))
    print("Model loaded")
    
    resnet_model.eval()
    
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for eval_idx, (inputs, labels) in enumerate(test_dataloader):
            outputs = resnet_model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (eval_idx+1)%45 == 0:
                print(eval_idx)
                break
    

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Model testing accuracy : {accuracy}")
    print(all_predictions)
    print(all_labels)

# %% Data Loader STN + training/testing

if __name__ == "__main__" and ( [cmd_input for cmd_input in ["test", "STNClassifier_model_training", "STNClassifier_model_testing"] if cmd_input in sys.argv]): # STNClassifier model class 

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

if __name__ == "__main__" and ( "data_loader" in sys.argv ) and ( ("STNClassifier_model_training" in sys.argv) or ("STNClassifier_model_testing" in sys.argv) ): # STNClassifier dataloader
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.225])
        ])
    dataset = datasets.ImageFolder(root="D:\Machine Learning\Chinese project\handwritten chinese numbers\dataloader_data", 
                                         transform=transform)
    
    dataset_length = len(dataset)
    print(dataset_length)
    train_dataset_length = round(dataset_length * 0.5)
    test_dataset_length = dataset_length - train_dataset_length
    train_data, test_data = random_split(dataset=dataset, lengths=[train_dataset_length, test_dataset_length])
    
    dataloader = DataLoader(train_dataset, batch_size=75*5, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)

if __name__ == "__main__" and ( ("test" in sys.argv) or ("STNClassifier_model_training" in sys.argv) ): # STNClassifier training

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    batch_size = z
    num_classes = len(np.unique(train_labels))
    
    STNClassifier_model = STNClassifier(batch_size=batch_size, in_channels=in_channels, height=m, width=n, num_classes=num_classes)
    optimizer = torch.optim.Adam(STNClassifier_model.parameters(), lr=0.001) 
    #100epochs lr=0.0001 89% lr=0.01 78% lr=0.001 90% 
    #200epochs lr=0.001 92%
    
    print(STNClassifier_model)
    total_params = sum(p.numel() for p in STNClassifier_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:_}")
    
    running_loss = 0
    epochs = int(input("Number of epochs : "))
    for epoch in tqdm(range(epochs)):
        STNClassifier_model.train()
        optimizer.zero_grad()
        output = STNClassifier_model(train_data)
        loss = F.nll_loss(output, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

    torch.save(STNClassifier_model.state_dict(), f"STNClassifier_model_{epochs}.pth")
    print("Finished Training, model saved")

if __name__ == "__main__" and ( ("test" in sys.argv) or ("STNClassifier_model_testing" in sys.argv) ): # STNClassifier testing
    
    if ("STNClassifier_model_training" not in sys.argv):
        epochs = int(input("Model's number of epochs to load (choose among [200, 500]): "))

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    num_classes = len(np.unique(train_labels))
    
    STNClassifier_model = STNClassifier(batch_size=z, in_channels=in_channels, height=m, width=n, num_classes=num_classes)
    STNClassifier_model.load_state_dict(torch.load(f"STNClassifier_model_{epochs}.pth"))
    
    STNClassifier_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = STNClassifier_model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT TERMINATED SUCCESSFULLY\033[0m")
