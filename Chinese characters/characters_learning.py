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
        array_list.append(np.array(cv2.imread(path + "/data/" + ImageListStack_image)))
        
    return np.stack(array_list, axis=2)

if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("tensor" in sys.argv) ):     # Generates tensor data only if no terminal argv, or "tensor" argument

            image_list = ImageListGenerator(reduced_suite_id, reduced_sample_id, reduced_code)
            stacked_images_array = ImageListStack(data_folder_path, image_list)

            train_data_tensor = torch.from_numpy(stacked_images_array).to(torch.float32)
            train_data_tensor = torch.nn.functional.normalize(train_data_tensor, dim=1)     # Norme 1
            torch.save(train_data_tensor, "train_data_tensor.pt")
            
            print("dtype | size:", train_data_tensor.dtype, "|", train_data_tensor.size())
            print("Train data mean, max :", torch.mean(train_data_tensor), torch.max(train_data_tensor))
 
# %%
import torch.nn as nn

class SimpleNN_1(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()        # Call __init__ contructor of torch.nn class
        self.layer1 = nn.Linear(64 * 64, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class SimpleNN_2(nn.Module):
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
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



if __name__ == "__main__" and ( (len(sys.argv) <= 1) or ("model" in sys.argv) ):  # Run model only if no terminal argv, or "model" argument        
        
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    print(f"Using {device} device")
    
    numbersNN_2 = SimpleNN_2()
    numbersNN_2_model = SimpleNN_2().to(device)
    print(numbersNN_2_model)

    input_array = input("Select image number to predict : ") 
    train_data = torch.load(current_folder_path + "/train_data_tensor.pt")[:,:, int(input_array)]
    logits = numbersNN_2_model(train_data)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
    
    plt.imshow(torch.numpy(train_data))
    print(torch.numpy(train_data))
    plt.title(f"Predicted class : {y_pred}")
    plt.show()
    #plt.sleep(2)
    plt.close('all')
    
    # numbersNN_1 = SimpleNN_1()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # num_epochs = 5

    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels = data

    #         optimizer.zero_grad()

    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

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

print(len(sys.argv))
print(__file__[__file__.rindex("\\")+1:], f"says : \033[1mscript ended successfully\033[0m")