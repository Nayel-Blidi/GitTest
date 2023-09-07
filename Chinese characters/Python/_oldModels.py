
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
import os
import sys

import datasetGenerator as dg

current_folder_path = os.path.dirname(os.path.abspath(__file__))
tensor_list = ["raw_tensor", "contrasted_tensor", "convoluted_tensor", "rotated_tensor"]
model_list = ["supervised_model", "convolutional_model"]
argv_list = tensor_list + model_list
for arg in sys.argv[1:]:
    if arg not in argv_list:
        raise ValueError(f"Wrong arg, try none or one of these :\n{argv_list}")

# %%
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
        
# %%
import torch.nn as nn

class SupervisedSimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SupervisedSimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.layer1 = nn.Linear(hidden_size, hidden_size//2)
        self.layer2 = nn.Linear(hidden_size//2, hidden_size//4)
        
        self.output_layer = nn.Linear(hidden_size//4, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.output_layer(x)
        return x

class ConvolutionalNN(nn.Module):
    def __init__(self, num_filters=16, in_channels=1, num_classes=10, kernel_size=3):
        super(ConvolutionalNN, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(3, 4)
        
        self.fc1 = nn.Linear(num_filters*16*16, num_classes)  
        # self.fc2 = nn.Linear(num_filters*4*4, num_classes)  

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

    #Flattens along dim>=1
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("supervised_model" in sys.argv) ):  
    # Runs model only if no terminal argv, or "supervised_model" argument        

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()
    
    input_size = m * n
    hidden_size = 128
    output_size = len(np.unique(targets))
    print(output_size)
    
    # train_data = train_data.permute(2, 0, 1)
    train_data = train_data.view(z, m*n)
    # print(train_data.size())
    # test_data = test_data.permute(2, 0, 1)
    test_data = test_data.view(z, m*n)
    # print(test_data.size())

    supervised_model = SupervisedSimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(supervised_model.parameters(), lr=0.01)
    print(supervised_model)
    total_params = sum(p.numel() for p in supervised_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Supervised model training
    num_epochs = int(input("Number of epochs : "))
    batch_size = z
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0

        for i in range(0, len(train_data), batch_size):
            inputs = train_data[i:i+batch_size]
            labels = train_labels[i:i+batch_size]

            optimizer.zero_grad()

            outputs = supervised_model(inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    print("Final loss :", running_loss)
    
    # Supervised model evaluation 
    supervised_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for i in (range(0, len(test_data), batch_size)):
            inputs = test_data[i:i+batch_size]
            labels = test_labels[i:i+batch_size]

            outputs = supervised_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Model testing
    print(np.unique(predicted.numpy(), return_counts=True))
    print(np.unique(test_labels.numpy(), return_counts=True))
    
    print(f"Accuracy: {100 * correct / total:.2f}%")
    
    # print(predicted.numpy()[test_labels.numpy() == 0])
    # print("Specific value prediction :\n", np.unique(predicted.numpy()[test_labels.numpy() == 0], return_counts=True))
    

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convolutional_model" in sys.argv) ):  
    # Runs model only if no terminal argv, or "convolutional_model" argument     

    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    train_data, train_labels, test_data, test_labels, (z, in_channels, m, n) = dg.TensorToTensors()

    
    num_classes = len(np.unique(targets))
    num_filters = int(input("Number of additional filters : "))

    convolutional_model = ConvolutionalNN(num_classes=len(np.unique(test_labels)), num_filters=num_filters)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(convolutional_model.parameters(), lr=0.1) #momentum=0.9
    print(convolutional_model)
    total_params = sum(p.numel() for p in convolutional_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
        
    num_epochs = int(input("Number of epochs : "))    
    for epoch in tqdm(range(num_epochs)):
        convolutional_model.train() 
        running_loss = 0.0

        optimizer.zero_grad()
        outputs = convolutional_model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

    convolutional_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = convolutional_model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()

    print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
    

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT ENDED SUCCESSFULLY\033[0m")