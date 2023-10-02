import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import os
import sys

from tqdm import tqdm


current_folder_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__": # Dataset cleaning
    # Training datatset cleaning
    train_data = pd.read_csv(f"{current_folder_path}/train.csv")
    # Getting rid of too specific string values
    train_data = train_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    
    # Getting rid of NaN values among the remaining columns
    # nan_count = train_data.isna().sum().sum()
    # print(nan_count)
    train_data.dropna(axis=0, inplace=True)
    print(train_data.head())

    train_labels = train_data[["Survived"]]
    
    train_features = train_data.drop(columns=["Survived"])
    train_features = pd.get_dummies(train_features)
    

    # Testing datatset cleaning
    test_data = pd.read_csv(f"{current_folder_path}/test.csv")
    answer = pd.read_csv(f"{current_folder_path}/gender_submission.csv")[["Survived"]]
    test_data[["Survived"]] = answer[["Survived"]]

    # Getting rid of too specific string values
    test_data = test_data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    
    # Getting rid of NaN values among the remaining columns
    test_data.dropna(axis=0, inplace=True)
    print(test_data.head())
    
    test_labels = test_data[["Survived"]]
    
    test_features = test_data.drop(columns=["Survived"])
    test_features = pd.get_dummies(test_features)
    



    print(test_data.head())
    
    print(test_features.tail())
    print(test_labels.tail())


if __name__ == "__main__" and "show_data" in sys.argv: #Prints datasets info
    print("\n train_features \n", train_features.head(1), "\n train_labels \n", train_labels.head(1))
    print("\n test_features \n", test_features.head(1), "\n test_labels \n", test_labels.head(1))
    # print(train_features.describe, train_labels.describe)
# %% NN model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.hidden_size= hidden_size
        self.output_size = output_size
        
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.layer1 = nn.Linear(hidden_size, hidden_size//2)
        self.layer2 = nn.Linear(hidden_size//2, hidden_size//4)
        
        self.batchnorm1 = nn.BatchNorm1d(hidden_size//2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size//4)

        self.output_layer = nn.Linear(hidden_size//4, output_size)
        self.sigmoid = nn.Sigmoid()  
        self.dropout = nn.Dropout1d(p=0.1)
            
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        
        x = self.dropout(x)
        
        x = self.output_layer(x)  
        #x = self.relu(x)    
        x = self.sigmoid(x)
        return x        

if __name__ == "__main__" and "tensor" in sys.argv:
    train_features = torch.from_numpy(train_features.to_numpy()).float()
    train_labels = torch.from_numpy(train_labels.to_numpy()).float()
    test_features = torch.from_numpy(test_features.to_numpy()).float()
    test_labels = torch.from_numpy(test_labels.to_numpy()).float()
    
    train_features = F.normalize(train_features)
    test_features = F.normalize(test_features)

print(train_features.size(), train_labels.size(), test_features.size(), test_labels.size())
# print(pd.DataFrame(train_features.numpy()).head(50))

if __name__ == "__main__" and "train" in sys.argv:

    m, n = train_features.shape
    input_size = n
    print(input_size)
    hidden_size = 64
    output_size = 1
    
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimize = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    model.train() 
    running_loss = 0.0
    
    losses_list = []
    num_epochs = int(input("Number of epochs : "))    
    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        outputs = model(train_features)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        losses_list.append(loss.item())

    print(f"Epoch {epoch+1}, Loss: {running_loss}")
    print(np.round(losses_list[::2], 6))

if __name__ == "__main__" and "test" in sys.argv:

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(test_features) 
        print(outputs[0:10].T)
        #predicted = torch.round(outputs.data)
        #_, predicted = torch.max(outputs.data, 1)
        predicted = torch.round(outputs.data)
        # predicted = np.round(outputs.numpy())
        total += test_labels.size(0)
        # correct += (predicted.numpy() == test_labels.numpy().T).sum().item()
        correct += (predicted == test_labels).sum().item()        
        
        print(predicted.numpy()[0:10].T, test_labels.numpy().T[0, 0:10])
        #correct += (predicted == test_labels.numpy().T[0]).sum().item()
        print(total, correct)

        
    # print(np.unique(predicted.numpy(), return_counts=True))
    # print(np.unique(test_labels.numpy(), return_counts=True))
    # print(predicted.numpy()[0:10], test_labels.numpy().T[0, 0:10])
    
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")



