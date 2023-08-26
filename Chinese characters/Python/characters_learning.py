# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch

from tqdm import tqdm
import os
import sys
import time


# %% Data loading
data_folder_path = "D:/Machine Learning/Chinese project/handwritten chinese numbers"
current_folder_path = os.path.dirname(os.path.abspath(__file__))

data_csv = pd.read_csv(data_folder_path+"/chinese_mnist.csv")
DataFrame_csv = pd.DataFrame(data_csv)
#reduced_DataFrame_csv = DataFrame_csv[DataFrame_csv['value' <= 9]] 

keys = DataFrame_csv.columns

# %% COLUMNS
suite_id = DataFrame_csv[keys[0]].values
sample_id = DataFrame_csv[keys[1]].values
code = DataFrame_csv[keys[2]].values
value = DataFrame_csv[keys[3]].values
value[value == 100 ] = 11
value[value == 1000] = 12
value[value == 10000] = 13
value[value == 100e6] = 14
print(np.unique(value))

# reduced_suite_id = reduced_DataFrame_csv[keys[0]].values
# reduced_sample_id = reduced_DataFrame_csv[keys[1]].values
# reduced_code = reduced_DataFrame_csv[keys[2]].values
# reduced_value = reduced_DataFrame_csv[keys[3]].values

# %% IMAGES TO ARRAY
def ImageListGenerator(suite_id, sample_id, code):
    
    image_list = []
    for ImageListGenerator_index, ImageListGenerator_id in enumerate(suite_id):
        image_list.append(f"input_{ImageListGenerator_id}_{sample_id[ImageListGenerator_index]}_{code[ImageListGenerator_index]}.jpg")

    return image_list

def ImageListStack(path, image_list):
    
    array_list = []
    for ImageListStack_index, ImageListStack_image in enumerate(image_list):
        array_list.append(np.array(cv2.imread(path + "/data/" + ImageListStack_image, cv2.IMREAD_GRAYSCALE)))
        
    return np.stack(array_list, axis=0)

def ConvolutedImageListStack(path, image_list):
    
    array_list = []
    for ImageListStack_index, ImageListStack_image in enumerate(image_list):
        array_list.append(np.array(cv2.imread(path + "/convoluted data/" + ImageListStack_image, cv2.IMREAD_GRAYSCALE)))
        
    return np.stack(array_list, axis=0)

def DatasetToTensor(data_folder_path=data_folder_path, suite_id=suite_id, sample_id=sample_id, code=code):
    
    image_list = ImageListGenerator(suite_id, sample_id, code)
    stacked_images_array = ImageListStack(data_folder_path, image_list)
    
    data_tensor = torch.from_numpy(stacked_images_array).to(torch.float32)
    data_tensor = data_tensor/255    # Normalized pixels
    torch.save(data_tensor, "data_tensor.pt")
    
    return data_tensor    

def DatasetToConvolutedTensor(suite_id=suite_id, sample_id=sample_id, code=code):
    
    image_list = ImageListGenerator(suite_id, sample_id, code)
    stacked_images_array = ImageListStack(data_folder_path, image_list)
    
    data_tensor = torch.from_numpy(stacked_images_array).to(torch.float32)
    data_tensor = data_tensor/255    # Normalized pixels
    torch.save(data_tensor, "data_tensor.pt")
    
    return data_tensor

def TensorToModelTensors():
    current_folder_path= os.path.dirname(os.path.abspath(__file__))
    data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    
    #Splitting the dataset in half, between train and test examples
    train_data = data_tensor[0::2, :,:]
    train_labels = torch.from_numpy(value[0::2]).to(torch.long)
    test_data = data_tensor[1::2, :,:]
    test_labels = torch.from_numpy(value[1::2]).to(torch.long)
    z, m, n = train_data.size()
    print("train_data tensor shape :", train_data.size())
    
    return train_data, train_labels, test_data, test_labels, (z, m, n)

# %%
if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("plain_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    data_tensor = DatasetToTensor()
    train_data, train_labels, test_data, test_labels, (z, m, n) = TensorToModelTensors()
    
    print("dtype | size:", data_tensor.dtype, "|", data_tensor.size())
    print("Train data mean, max :", torch.mean(data_tensor), torch.max(data_tensor))
            
if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convoluted_tensor" in sys.argv) ):     
    # Generates tensor data only if no terminal argv, or "tensor" argument
    data_tensor = DatasetToConvolutedTensor()
    train_data, train_labels, test_data, test_labels, (z, m, n) = TensorToModelTensors()
    
    print("dtype | size:", data_tensor.dtype, "|", data_tensor.size())
    print("Train data mean, max :", torch.mean(data_tensor), torch.max(data_tensor))


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
        
        self.fc1 = nn.Linear(num_filters*16*16, num_filters*4*4)  
        self.fc2 = nn.Linear(num_filters*4*4, num_classes)  

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
    # Run model only if no terminal argv, or "supervised_model" argument        

    train_data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    #Splitting the dataset in half, between train and test examples
    train_data = train_data_tensor[0::2, :,:]
    train_labels = torch.from_numpy(value[0::2]).to(torch.long)
    test_data = train_data_tensor[1::2, :,:]
    test_labels = torch.from_numpy(value[1::2]).to(torch.long)
    z, m, n = train_data.size()
    print(train_data.size())
    
    input_size = m * n
    hidden_size = 128
    output_size = len(np.unique(value))
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
    
    print(f"Accuracy: {100 * correct / total}%")
    
    # print(predicted.numpy()[test_labels.numpy() == 0])
    # print("Specific value prediction :\n", np.unique(predicted.numpy()[test_labels.numpy() == 0], return_counts=True))
    

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convolutional_model" in sys.argv) ):  
    # Run model only if no terminal argv, or "convolutional_model" argument     

    train_data_tensor = torch.load(current_folder_path + "/data_tensor.pt")
    #Splitting the dataset in half, between train and test examples
    train_data = train_data_tensor[0::2, :,:].unsqueeze(1)
    train_labels = torch.from_numpy(value[0::2]).to(torch.long)
    test_data = train_data_tensor[1::2, :,:].unsqueeze(1)
    test_labels = torch.from_numpy(value[1::2]).to(torch.long)
    z, in_channels, m, n = train_data.size()
    print(train_data.size())
    
    num_classes = len(np.unique(value))
    num_filters = int(input("Number of additional filters : "))

    criterion = nn.CrossEntropyLoss()
    convolutional_model = ConvolutionalNN(num_classes=len(np.unique(test_labels)), num_filters=num_filters)
    optimizer = torch.optim.Adam(convolutional_model.parameters(), lr=0.1) #momentum=0.9

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

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    
     
# %%

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("deep_model" in sys.argv) ):
    # Run model only if no terminal argv, or "model" argument        
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"Using {device} device")
    
    DeepSimpleNN()
    DeepSimpleNN_model = DeepSimpleNN().to(device)
    print(DeepSimpleNN_model)

    input_array = input("Select image number to predict : ") 
    train_data = torch.load(current_folder_path + "/train_data_tensor.pt")[:,:, int(input_array)].flatten()

    logits = DeepSimpleNN_model(train_data)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
    # plt.imshow(torch.numpy(train_data))
    # plt.title(f"Predicted class : {y_pred}")
    # plt.show()
    #plt.sleep(2)
    #plt.close('all')
    

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT TERMINATED SUCCESSFULLY\033[0m")