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

"""
Training data folder structure:
- data
    - jpex eamples ()
"""

# %% Data loading
data_folder_path = "d:/Machine Learning/Chinese project/handwritten chinese numbers"
current_folder_path = os.path.dirname(os.path.abspath(__file__))

data_csv = pd.read_csv(data_folder_path+"/chinese_mnist.csv")
DataFrame_csv = pd.DataFrame(data_csv)
reduced_DataFrame_csv = DataFrame_csv[DataFrame_csv['value'] <= 9] #Reduced dataset, keeping numbers between 0 and 9

keys = DataFrame_csv.columns

# %% COLUMNS
suite_id = DataFrame_csv[keys[0]].values
sample_id = DataFrame_csv[keys[1]].values
code = DataFrame_csv[keys[2]].values
value = DataFrame_csv[keys[3]].values

reduced_suite_id = reduced_DataFrame_csv[keys[0]].values
reduced_sample_id = reduced_DataFrame_csv[keys[1]].values
reduced_code = reduced_DataFrame_csv[keys[2]].values
reduced_value = reduced_DataFrame_csv[keys[3]].values

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
        
    return np.stack(array_list, axis=2)

def ConvolutedImageListStack(path, image_list):
    
    array_list = []
    for ImageListStack_index, ImageListStack_image in enumerate(image_list):
        array_list.append(np.array(cv2.imread(path + "/data/" + ImageListStack_image, cv2.IMREAD_GRAYSCALE)))
        
    return np.stack(array_list, axis=2)

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("plain_tensor" in sys.argv) ):     # Generates tensor data only if no terminal argv, or "tensor" argument

            image_list = ImageListGenerator(reduced_suite_id, reduced_sample_id, reduced_code)
            stacked_images_array = ImageListStack(data_folder_path, image_list)

            train_data_tensor = torch.from_numpy(stacked_images_array).to(torch.float32)
            #train_data_tensor = torch.nn.functional.normalize(train_data_tensor, dim=1)     # Norme 1
            train_data_tensor = train_data_tensor/255    # Normalized pixels
            torch.save(train_data_tensor, "train_data_tensor.pt")
            
            print("dtype | size:", train_data_tensor.dtype, "|", train_data_tensor.size())
            print("Train data mean, max :", torch.mean(train_data_tensor), torch.max(train_data_tensor))
if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("convoluted_tensor" in sys.argv) ):     # Generates tensor data only if no terminal argv, or "tensor" argument

    image_list = ImageListGenerator(reduced_suite_id, reduced_sample_id, reduced_code)
    stacked_images_array = ImageListStack(data_folder_path, image_list)

    train_data_tensor = torch.from_numpy(stacked_images_array).to(torch.float32)
    #train_data_tensor = torch.nn.functional.normalize(train_data_tensor, dim=1)     # Norme 1
    train_data_tensor = train_data_tensor/255    # Normalized pixels
    torch.save(train_data_tensor, "train_data_tensor.pt")
    
    print("dtype | size:", train_data_tensor.dtype, "|", train_data_tensor.size())
    print("Train data mean, max :", torch.mean(train_data_tensor), torch.max(train_data_tensor))

    fig = plt.subplots(4, 5)
    plt.suptitle("Convoluted images")
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(train_data.numpy()[i,:].reshape(m,n))
        plt.title( f"Real : {test_labels.numpy()[i]}")
    plt.close('all')
    plt.show()
        

# %%
import torch.nn as nn

class SupervisedSimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SupervisedSimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class DeepSimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("supervised_model" in sys.argv) ):  
    # Run model only if no terminal argv, or "supervised_model" argument        

    train_data_tensor = torch.load(current_folder_path + "/train_data_tensor.pt")
    #Splitting the dataset in half, between train and test examples
    train_data = train_data_tensor[:,:, 0::2]
    train_labels = torch.from_numpy(reduced_value[0::2]).to(torch.long)
    test_data = train_data_tensor[:,:, 1::2]
    test_labels = torch.from_numpy(reduced_value[1::2]).to(torch.long)
    m, n, z = train_data.size()
    
    input_size = m * n
    hidden_size = 128
    output_size = len(np.unique(reduced_value))

    fig = plt.subplots(4, 5)
    plt.suptitle("Sliced train_data array")
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(train_data.numpy()[:,:,i])
        plt.title( f"Real : {test_labels.numpy()[i]}")
        
    train_data = train_data.permute(2, 0, 1)
    train_data = train_data.view(z, m*n)
    print(train_data.size())
    test_data = test_data.permute(2, 0, 1)
    test_data = test_data.view(z, m*n)
    print(test_data.size())

    fig = plt.subplots(4, 5)
    plt.suptitle("Flatten reshaped train_data array")
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(train_data.numpy()[i,:].reshape(m,n))
        plt.title( f"Real : {test_labels.numpy()[i]}")
    plt.close('all')
    plt.show()

    supervised_model = SupervisedSimpleNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(supervised_model.parameters(), lr=0.001)

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
            
        # if epoch//10 == 0:
        #     print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_data)}")
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
    
    print(predicted.numpy()[test_labels.numpy() == 0])
    print(np.unique(predicted.numpy()[test_labels.numpy() == 0], return_counts=True))
    
    fig = plt.subplots(4, 5)
    for i in range(20):
        plt.subplot(4, 5, i+1)
        plt.imshow(test_data.numpy()[i, :].reshape(m, n))
        plt.title( f"Predicted : {predicted.numpy()[i]}, \n Real : {test_labels.numpy()[i]}")
    plt.close('all')
    plt.show()
    
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
    

# %%

class DataHandler:
    
    def __init__(self, path, suite_id, sample_id, code, value):
        
        self.path = path
        self.suite_id = suite_id
        self.sample_id = sample_id
        self.code = code
        self.value = value

    def ImageShow(self, *args):
        
        plt.imshow(plt.imread(self.path + "/data" + f"/input_{args[0]}_{args[1]}_{args[2]}.jpg"))
        plt.show()
        
        return None


# dataClass = DataHandler(data_folder_path,x suite_id, sample_id, code, value)
# dataClass.ImageShow(1, 1, 1)

# for k in range(12):
#     plt.subplot(3, 4, k+1)
#     plt.imshow(plt.imread(data_folder_path + "/data" + f"/input_1_1_{k+1}.jpg"))
#     plt.title(k)
# plt.show()

print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mSCRIPT TERMINATED SUCCESSFULLY\033[0m")