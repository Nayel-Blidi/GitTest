# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: K,title,incorrectly_encoded_metadata,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
# ---

# %%
#================ PART II ================#

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from tqdm import tqdm
from math import trunc

import ML

#The cloud dataset, 6go, np arrays 30+go
#https://www.kaggle.com/datasets/christianlillelund/the-cloudcast-dataset?resource=download

#The cloud dataset : dataset of suboptimal npy arrays to dataarray

#Edited dataset after npy_to_dataarray.py computation
#Data is now structured in 2 datasets for learning :
    #merged_darray_labels regroups the masked array of every cloud images by labels
    #Shapes info :
        #Time : roughly 365*24*4/15 files, ie one image every 15min
        #Label : the type of cloud, defined by its altitude
        #x : longitude coordinate
        #y : lattitude coordinates
    
    #merged_darray_borders regroups the borders of every cloud images by labels
    #Shapes info :
        #Time : roughly 365*24*4/15 files, ie one image every 15min (0-)
        #Label : the type of cloud, defined by its altitude (0-12)
        #x : longitude coordinate (0-767)
        #y : lattitude coordinates (0-767)
    
"""
Specify the absolute path of the 'Project' folder location :
"""
abs_path = "D:/Machine Learning/Project"

path_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_borders = abs_path+"/TheCloudDataset_labelized_borders_train"

files = [os.path.join(path_labels, f) for f in os.listdir(path_labels) if f.endswith(".nc")]
array = np.load(abs_path+"/The cloud dataset/2017M01/0.npy")

# %%
fig, ax = plt.subplots(4,8)
for k in range(32):
    if k<=15:
        plt.subplot(4,8,k+1)
        plt.imshow(np.load(abs_path+f"/The cloud dataset/2017M01/{k*96}.npy"), cmap='cool')
        plt.title(f"{k+1}/01/2017")
    if k<=15:
        plt.subplot(4,8,k+17)
        plt.imshow(np.load(abs_path+f"/The cloud dataset/2018M01/{k*96}.npy"), cmap='cool')
        plt.title(f"{k+1}/01/2018")

plt.get_current_fig_manager().full_screen_toggle()
plt.savefig(abs_path+"/daily_img_2017_2018.png")
plt.close('all')

# %%
"""
The learning will follow these steps :
    1 - PCA over every image to reduce its size and thus computation time
    2 - 1vsAll algorithm over labels and borders
    2bis - 1vsAll testing with the 2018 dataset (might merge it as well)
    3 - NN over labels and borders
    3b - NN testing with the 2018 dataset (might merge it as well)
    4 - comparison of the algorithm results
    4b - testin of the two algorithm over raw input (live images from the EUMETSAT)
    5 - trying to improve the results by combining labels and borders when learning
    5b - testing over 2018 dataset
    5c - testing over raw input (live images from the EUMETSAT)
    
Depending on the difficulty, file size and computation time, every step may not be acheived, if not sucessfull.
"""

# %%
#================ 1 - PCA ================#

# %%

def slice_array(array, sclices=6):
    
    arr_blur_reduced = array[::sclices, ::sclices]
    
    return arr_blur_reduced
    

def pca_array(array, k_main_components, normalization=False):
    
    array_pca, pca_info = ML.pca(array, k_main_components, normalization)
    
    return array_pca, pca_info


def pca_recoverArray(array_pca, pca_info):
    
    array_recover = ML.pca_recoverData(array_pca, pca_info)
    
    return array_recover


def compute_pca_dataset(dataset, k_main_components=50):
    
    dataarray_pca = dataset['data'].values.astype(np.float32)
    len_time, len_label, m, n = dataarray_pca.shape
    
    m, n = m//6, n//6
    
    flattened_pca_array = np.zeros((len_label, len_time//8, m*k_main_components))
    for t in tqdm(range(0, len_time, 8)):
        for l in range(len_label):
            try:
                array_pca = pca_array(slice_array(dataarray_pca[t,l,:,:]), k_main_components)[0]
            except:
                array_pca = np.zeros((m, k_main_components))
            flattened_pca_array[l, t//8, :] = array_pca.flatten()
    
    return flattened_pca_array


def save_pca_flattened_dataset(path_in, path_out, k_main_components=50):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".nc")]

    for k in tqdm(range(len(files))):
        dataset = xr.open_dataset(f"{files[k]}")
        flattened_pca_array = compute_pca_dataset(dataset, k_main_components)
        flattened_pca_array = xr.DataArray(flattened_pca_array, dims=('time', 'data_pca', 'label'))
        flattened_pca_array.to_netcdf(f"{path_out}/pca_k{k_main_components}_{k}.nc")
    
    return None

# %% TEST COMPUTE PCA
dataset = xr.open_dataset(abs_path+"/TheCloudDataset_labelized_arrays_train/2017M01_concatenated_darray_separated_0.nc")
array_pca = compute_pca_dataset(dataset, k_main_components=10)
darray_pca = xr.DataArray(array_pca, dims=('time', 'data_pca', 'label'))

# %% TEST PCA&RECOVER ARRAY
array = xr.open_dataset(abs_path+"/TheCloudDataset_labelized_arrays_train/2017M01_concatenated_darray_separated_0.nc")['data'].values[0,0,:,:] 
array = slice_array(array)

array_pca, array_pca_info = pca_array(array, k_main_components=30, normalization=True)
array_pca_recovered = pca_recoverArray(array_pca, array_pca_info)

l = [array, array_pca, array_pca_recovered]
for k in range(3):
    plt.subplot(1,3,k+1)
    plt.imshow(l[k], cmap='gray')
plt.suptitle("Image 0 label 0 / PCA img / PCA img recovered")
plt.savefig(abs_path+"/1b_pca_example_k30.png")

# %% TEST PCA&RECOVERS BORDER
array = xr.open_dataset(abs_path+"/TheCloudDataset_labelized_borders_train/2017M01_concatenated_darray_separated_borders_0.nc")['data'].values[0,0,:,:] 

array_pca, array_pca_info = pca_array(array, k_main_components=50)
array_pca_recovered = pca_recoverArray(array_pca, array_pca_info)

l = [array, array_pca, array_pca_recovered]
for k in range(3):
    plt.subplot(1,3,k+1)
    plt.imshow(l[k], cmap='gray')

# %% PCA COMPUTATION ARRAYS
k_main_components = 20
path_in = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_out = abs_path+f"/TheCloudDataset_labelized_arrays_train_pca_k{k_main_components}"

#Dirty fonction calling to store the computed pca datasets
save_pca_flattened_dataset(path_in, path_out, k_main_components) 

# %% PCA COMPUTATION BORDERS
k_main_components = 10
path_in = abs_path+"/TheCloudDataset_labelized_borders_train"
path_out = abs_path+"/TheCloudDataset_labelized_borders_train_pca_k{k_main_components}"

#Dirty fonction calling to store the computed pca datasets
save_pca_flattened_dataset(path_in, path_out, k_main_components)    

# %%
"""                   
Over the extremely high complexity and time consumption of the computation, choices had to be made :
(pca computation and writing duration was ~6h)
    1) The PCA had to be very rough, with 10 k main components selected (might try 20-30 afterwards)
    2) Data had to be reduced again : instead of 15min slices, it's now 2 hours. The impact should be small
    enough over the futur learning, and it's still better than taking instead only 2.5 months of the whole year
    instead (for the same data weight reduction) and thus missing the seasons variations
    3) Data is now stored as non-compressed float 32, which should speed up the learning computation
    4) Data is still stored as individual daily arrays, as merging is still to heavy :
        - As for the previous compression, there are still too many files to load in the memory
        - Lazy computation may work better on uncompressed data, but might be counter effective when
        computing the neural network learning.
"""

# %%
#================ 2 - OneVsAll ALGORITHM ================#

# %%

def data_toarray_selection(path_in, variable, sample_size=324, num_labels=13):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".nc")]
    
    #To check the length of a vector of an image (may change depending on the PCA)
    data_info = xr.open_dataset(files[0])[variable].values
    len_data = data_info.shape[-1]
    len_data_daily = data_info.shape[-2]
    
    y = np.zeros((sample_size*len_data_daily))
    X = np.zeros((sample_size*len_data_daily, len_data))
    
    print(X.shape)
    
    for t in tqdm(range(sample_size*len_data_daily)):
        X[t, :] = xr.open_dataset(files[t//len_data_daily])[variable].values[t%num_labels, t%len_data_daily, :].flatten()
        y[t] = t%num_labels
    
    return X, y


def OneVsAll_dataset(path_in, variable, sample_size=324, lambda_=1, num_labels=13, itterations=10):
    
    print("Loading of X and y")
    X, y = data_toarray_selection(path_in, variable, sample_size, num_labels)
    print("X and y finished loading")
    
    trained_theta = ML.oneVsAll(X, y, num_labels, lambda_, tol=1e-3)[1]
    
    return trained_theta, X, y


def OneVsAll_testDataset_pca(path_in, trained_theta, variable, sample_size=30, num_labels=13):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".nc")]
    
    #To check the length of a vector of an image (may change depending on the PCA)
    data_info = xr.open_dataset(files[0])[variable].values
    len_data = data_info.shape[-1]
    len_data_daily = data_info.shape[-2]
    
    X = np.zeros((sample_size*len_data_daily, len_data))
    for t in tqdm(range(sample_size*len_data_daily)):
        X[t, :] = xr.open_dataset(files[t//len_data_daily])[variable].values[t%num_labels, t%len_data_daily].flatten()
    
    p = ML.predictOneVsAll(trained_theta[:,1:], X)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, sample_size*len_data_daily, num_labels):
            if p[t] == k:
                success_of_label_k+=1
        success_of_label_k = success_of_label_k / (sample_size*len_data_daily/num_labels)
        print("Success of label", k, "is : ", success_of_label_k*100, "%")
    return p


def OneVsAll_testArray(path_in, trained_theta, variable, sample_size=30, num_labels=13, k_main_components=10, slices=6):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".npy")]
    size_array_pca = np.load(files[0]).shape[0]*k_main_components
    
    X = np.zeros((sample_size*num_labels, size_array_pca//slices))
    for t in tqdm(range(sample_size*num_labels)):
        array = np.load(files[t//num_labels])
        array[array != (t+1)%num_labels] = 0
        array[array == (t+1)%num_labels] = 1
        array_pca = pca_array(slice_array(array), k_main_components)[0]
        X[t, :] = array_pca.flatten()
    
    p = ML.predictOneVsAll(trained_theta[:,1:], X)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, sample_size*num_labels, num_labels):
            if p[t] == k%num_labels:
                success_of_label_k+=1
        success_of_label_k = success_of_label_k / sample_size
        print("Success of label", k, "is : ", success_of_label_k*100, "%")
    return p


def plot_OneVsAll_testDataset(path_in, trained_theta, p, sample_size=30, subplot_size=3, num_labels=13):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".nc")]
    
    len_subplot = subplot_size**2
    
    fig, ax = plt.subplots(subplot_size, subplot_size)
    fig.subplots_adjust(hspace=0.5)
    for t in tqdm(range(len_subplot)):
        array = xr.open_dataset(files[t//num_labels])['data'].values[t//num_labels, t%num_labels, :, :]
        
        try:
            plt.subplot(subplot_size, subplot_size, t+1)
            plt.imshow(array)
            plt.title(f"Real label is : {t%num_labels} \n Predicted label is : {p[t]}")
        except:
            break
        
    plt.savefig(abs_path+f"/OneVsAll_2017_k10.png")

    return None

# %% 1VSALL ARRAYS K=10
path_in_arrays_pca = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k10"
#If dataset is PCA, data variable is '__xarray_dataarray_variable__', 'data' otherwise
variable = "__xarray_dataarray_variable__"

trained_theta_arrays, X_arrays, y_arrays = OneVsAll_dataset(path_in_arrays_pca, variable, sample_size=324,
                                                            lambda_=1, num_labels=13, itterations=30)

np.save(abs_path+"/trained_theta_k20_itt30_sample324.npy", trained_theta_arrays)


# %% 1VSALL ARRAYS K=20
path_in_arrays_pca = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k10"
#If dataset is PCA, data variable is '__xarray_dataarray_variable__', 'data' otherwise
variable = "__xarray_dataarray_variable__"

trained_theta_arrays, X_arrays, y_arrays = OneVsAll_dataset(path_in_arrays_pca, variable, sample_size=324,
                                                            lambda_=1, num_labels=13, itterations=30)

p_arrays = OneVsAll_testDataset_pca(path_in_arrays_pca, trained_theta_arrays, variable,
                                    sample_size=1, num_labels=13)

plot_OneVsAll_testDataset(path_in_arrays_pca, trained_theta_arrays, p_arrays, variable, 
                          sample_size=30, subplot_size= 6, num_labels=13)

# %%
"""
New choices had to be made : even when reducing PCA to 10 main components vectors, the output array
size is 728*10=7280 "pixels" to study, which way too long for scipy.optimize.minimize.
Even when sampling ony the first month, the learning matrix was 360*7280.
Thus, we had a new idea : to first reduce the images size before computing both the learning 
algorithm and the PCA of the data.
Thus, for 10 main components, thetas calculation was ~16min, and 1h10min for k=20.
In any way, we still mainly hope for the to-be-done neural network to converge more quickly and be
able to handle all the data

PS : array slicing gave bad results on the convoluted border dataset. Convolution product should 
be done over the newly sliced data.
PS2 : tol in scipy.optimize.minimize was set to 10e-3 instead of the default 10e-8 considering 
the values of the arrays being aroung 10e1
"""

# %%
#================ 2b - OneVsAll ALGORITHM TESTING ================#

# %% TESTING K=10 2017 (TRAINED)
trained_theta_arrays = np.load(abs_path+"/trained_theta_k10_itt30_sample324.npy")
path_in_arrays_pca = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k10"
variable = "__xarray_dataarray_variable__"

p_arrays = OneVsAll_testDataset_pca(path_in_arrays_pca, trained_theta_arrays, variable,
                                sample_size=324, num_labels=13)

path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train"

plot_OneVsAll_testDataset(path_in_arrays, trained_theta_arrays, p_arrays, 
                          sample_size=324, subplot_size=4, num_labels=13)

# %% TESTING K=20 2017 (TRAINED)
trained_theta_arrays = np.load(abs_path+"/trained_theta_k20_itt30_sample324.npy")
path_in_arrays_pca = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k20"
variable = "__xarray_dataarray_variable__"

p_arrays = OneVsAll_testDataset_pca(path_in_arrays_pca, trained_theta_arrays, variable,
                                sample_size=324, num_labels=13)

path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train"

plot_OneVsAll_testDataset(path_in_arrays, trained_theta_arrays, p_arrays, 
                          sample_size=324, subplot_size=4, num_labels=13)

# %% TESTING K=10 2018 (TEST)
trained_theta_arrays = np.load(abs_path+"/trained_theta_k10_itt30_sample324.npy")
path_in_arrays_2018 = abs_path+"/The cloud dataset/2018M10"
variable = "__xarray_dataarray_variable__"

p_arrays = OneVsAll_testArray(path_in_arrays_2018, trained_theta_arrays, variable,
                                sample_size=1000, num_labels=13, k_main_components=10)

# %% TESTING K=20 2018 (TEST)
trained_theta_arrays = np.load(abs_path+"/trained_theta_k20_itt30_sample324.npy")
path_in_arrays_2018 = abs_path+"/The cloud dataset/2018M11"
variable = "__xarray_dataarray_variable__"

p_arrays = OneVsAll_testArray(path_in_arrays_2018, trained_theta_arrays, variable,
                                sample_size=500, num_labels=13, k_main_components=20)

# %%
"""
Testing over the whole 2017 folder :
When testing with k=20 over the learning dataset, results are excellent exepted for 3 categories :
labels 2, 3 and 8. This tremendous lack of performance comes from a lack of data, as occurences of 
those labels are very rare. They then should be removed from the learning examples, or be trained on 
a dedicated dataset, with more examples (most of the arrays are null)

PS : error calculation is sometimes slightly above 100%, depending on the rounding of the size of the folder

Testing over the whole January 2018 folder :
When testing with k=10 and k=20 over testing dataset, results are overwhelmingly disapointing.
Actually, it recognizes consistantly some labels but doesn't attribute it the right value. 
(eg : label 4 is always recognised as 0)
The testing has to be realised over larger sets to confirm the biaises. If confirmed, then the finesse
of the algorithm should be reduced, to help breaking those biaises.


Testing over the whole 2018 folder :
Biaises seems to be the same. Leet's keep going by woorking on the neural network before coming back
here to merge some categories. 
"""

# %%
#================ 3 - NEURAL NETWORK ================#

# %%

def run_datasetNN(path_in, variable="__xarray_dataarray_variable__", sample_size=100, hidden_layer_size=32, num_labels=13, itterations=100):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".nc")]
    
    #To check the length of a vector of an image (may change depending on the PCA)
    data_info = xr.open_dataset(files[0])[variable].values
    len_data = data_info.shape[-1]
    len_data_daily = data_info.shape[-2]
    
    X, y = data_toarray_selection(path_in, variable, sample_size, num_labels)
    
    Theta1, Theta2 = ML.nnOneLayer(X, y, hidden_layer_size, num_labels, itterations)
        
    return Theta1, Theta2


def test_datasetNN(path_in, Theta1, Theta2, sample_size=100, num_labels=13):
    
    X, y = data_toarray_selection(path_in, variable, sample_size, num_labels)
    p_arrays = ML.predOneLayer(X, Theta1, Theta2)
    
    len_sample, len_data = X.shape
    print(X.shape)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, len_sample, num_labels):
            if p_arrays[t] == k:
                success_of_label_k+=1
        success_of_label_k = trunc(success_of_label_k/(len_sample/num_labels)*10000)
        print("Success of label", k, "is : ", success_of_label_k/100, "%")

    return p_arrays


def test_arrayNN(path_in, Theta1, Theta2, k_main_components=10, sample_size=100, num_labels=13):
    
    files = [os.path.join(path_in, f) for f in os.listdir(path_in) if f.endswith(".npy")]
    size_array_pca = np.load(files[0]).shape[0]*k_main_components
    
    X = np.zeros((sample_size*num_labels, size_array_pca//6))
    for t in tqdm(range(sample_size*num_labels)):
        array = np.load(files[t//num_labels])
        array[array != (t+1)%num_labels] = 0
        array[array == (t+1)%num_labels] = 1
        array_pca = pca_array(slice_array(array), k_main_components)[0]
        X[t, :] = array_pca.flatten()
    
    p_arrays = ML.predOneLayer(X, Theta1, Theta2)

    len_sample, len_data = X.shape
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, len_sample, num_labels):
            if p_arrays[t] == k:
                success_of_label_k+=1
        success_of_label_k = trunc(success_of_label_k/(len_sample/num_labels)*10000)
        print("", "Success of label", k, "is : ", success_of_label_k/100, "%")

    return p_arrays, X

# %%
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k10"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = run_datasetNN(path_in_arrays, variable, sample_size=324, num_labels=13, itterations=100)

np.save(abs_path+"/trained_NN_k10_itt100_sample324.npy", np.array([Theta1, Theta2], dtype=object))

# %%
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k20"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = run_datasetNN(path_in_arrays, variable, sample_size=324, num_labels=13, itterations=100)

np.save(abs_path+"/trained_NN_k20_itt100_sample324.npy", np.array([Theta1, Theta2], dtype=object))

# %%
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k30"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = run_datasetNN(path_in_arrays, variable, sample_size=324, num_labels=13, itterations=100)

np.save(abs_path+"/trained_NN_k30_itt100_sample324.npy", np.array([Theta1, Theta2], dtype=object))

# %%
#================ 3b - NEURAL NETWORK TESTING ================#

# %% TESTING K=10 2017 (TRAINED)
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k10"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = np.load(abs_path+"/trained_NN_k10_itt100_sample324.npy", allow_pickle=True)

p_arrays = test_datasetNN(path_in_arrays, Theta1, Theta2, sample_size=324, num_labels=13)

# %% TESTING K=20 2017 (TRAINED)
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k20"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = np.load(abs_path+"/trained_NN_k20_itt100_sample324.npy", allow_pickle=True)

p_arrays = test_datasetNN(path_in_arrays, Theta1, Theta2, sample_size=324, num_labels=13)

# %% TESTING K=30 2017 (TRAINED)
path_in_arrays = abs_path+"/TheCloudDataset_labelized_arrays_train_pca_k30"
variable = "__xarray_dataarray_variable__"

Theta1, Theta2 = np.load(abs_path+"/trained_NN_k20_itt100_sample324.npy", allow_pickle=True)

p_arrays = test_datasetNN(path_in_arrays, Theta1, Theta2, sample_size=324, num_labels=13)

# %% TESTING K=10 2018 (TEST)
path_in_arrays = "D:/The cloud dataset/2018M01"

Theta1, Theta2 = np.load("D:/Machine Learning/Project/trained_NN_k10_itt100_sample324.npy", allow_pickle=True)

p_arrays, X = test_arrayNN(path_in_arrays, Theta1, Theta2, k_main_components=10, sample_size=1000,  num_labels=13)

# %% TESTING K=20 2018 (TEST)
path_in_arrays = abs_path+"/The cloud dataset/2018M01"

Theta1, Theta2 = np.load(abs_path+"/trained_NN_k20_itt100_sample324.npy", allow_pickle=True)

p_arrays = test_arrayNN(path_in_arrays, Theta1, Theta2, k_main_components=20, sample_size=1000,  num_labels=13)

# %%
#================ 4 - RESULTS ANALYSIS ================#

# %%
"""
The results are not satisfying enough to keep going. Some labels are (kinda) working whereas others
are fuly broken (for multiple reasons).
The way of improvement are as listed here :
    1) Increasing the resolution of the PCA is not an option for the 1vsAll algorithm, as the maximum 
    computation time has already been reached. Altough, larger k main components values can be computed 
    for the Neural Network
    2) Merging some labels can be planned, as clouds of a close categories are mainly flying in a pack
    (cf images 0 and 1), and can add weight to a categorie when compared to an other.
    3) Using the borders convolution to improve the accuracy, or maybe be the only dataset to learn from.
"""
# %%
#================ END OF PART II ================#

# %%
    






