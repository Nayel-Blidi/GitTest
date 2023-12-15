# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
# ---

# %%
#================ PART III ================#

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from tqdm import tqdm
from scipy.signal import convolve2d
from natsort import natsorted
from math import trunc

import ML

#The cloud dataset, 6go, np arrays 30+go
#https://www.kaggle.com/datasets/christianlillelund/the-cloudcast-dataset?resource=download
#https://vision.eng.au.dk/cloudcast-dataset/

#The cloud dataset : dataset of suboptimal npy arrays to dataarray

#Data is structured in 24 files, for every month, named {year}M{month}
#Every npy is the labelized cloud category, at t={num_array}*15 minutes
#The data is studied above europe, centered on France, and covers some parts of the Maghreb
#More especially between the long and lats : cf GEO.npz

"""
Specify the absolute path of the 'Project' folder location :
"""
abs_path = "D:/Machine Learning/Project"

# %%
#================ CREATING (again) A NEW DATASET ================#

# %%
"""
Let's merge the dataset by larger categories, as adjacent clouds usually belong to the same category
"""

# %%
fig, ax = plt.subplots(4,6)
fig.subplots_adjust(hspace=0.4)
for k in range(0, 96*4*6, 96):
    array = np.load(f"D:/The cloud dataset/2017M01/{k}.npy")
    plt.subplot(4,6, k//96+1)
    plt.imshow(array, cmap='cool')
    plt.title(f"{k//96+1} January 2017")
#plt.get_current_fig_manager().full_screen_toggle()
#plt.savefig("D:/Machine Learning/Project/10_daily_img_January_2017.png")
#plt.close('all')

# %%
array = np.load(abs_path+"/The cloud dataset/2017M01/0.npy")
plt.imshow(array, cmap='cool')

# %%

def slice_array(array, slices=6):
    """
    Divides the resolution of an input array by slices
    """
    return array[::slices, ::slices]

    
def pca_array(array, k_main_components, normalization=True):
    """
    Module fonction.
    Runs PCA algorithm on an input array and keep the first k_main_components.
    Input array can be normalized or not.
    """
    array_pca, pca_info = ML.pca(array, k_main_components, normalization)
    return array_pca, pca_info


def merged_arrays(file, num_labels=4, slices=6, k_main_components=30):
    """
    Merges all the arrays of a folder together, in 4 sub netcdf files :
        1) Original sliced input array
        2) PCA sliced input array
        3) Convoluted sliced input array
        4) PCA convoluted sliced array
    """
    array = np.load(file)
    array = slice_array(array, slices)
    
    m, n = array.shape
    kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
    
    new_array = np.zeros((num_labels, m, n))
    new_array_pca = np.zeros((num_labels, m, k_main_components))
    for k in range(num_labels):
        array_label = np.zeros((m,n))
        array_label[array==k*3+1] = 1
        array_label[array==k*3+2] = 1
        array_label[array==k*3+3] = 1
        if k==0:
            array_label[array==k] = 1 

        new_array[k, :, :] = array_label
        new_array_pca[k,:,:] = pca_array(array_label, k_main_components)[0]
    
    new_array_border = np.zeros((num_labels, m, n))
    new_array_border_pca = np.zeros((num_labels, m, k_main_components))
    for k in range(num_labels):
        new_array_border[k,:,:] = convolve2d(new_array[k,:,:], kernel, mode='same', boundary='symm')
        new_array_border_pca[k,:,:] = pca_array(new_array_border[k,:,:], k_main_components)[0]
        
    return new_array, new_array_pca, new_array_border, new_array_border_pca


def save_dataset(path_in, path_out, year="2017", num_labels=4, slices=6, k_main_components=30, compression=False):
    """
    Save all the arrays in path_in folder into path_out folder, as merged datasets alongside dimension time.
    Calls the function merged_arrays
    """
    files = []
    for k in range(1, 13):
        if k <=9:
            file_names = natsorted(os.listdir(abs_path+f"/The cloud dataset/{year}M0{k}"))
            files += [os.path.join(f"{path_in}/{year}M0{k}", f) for f in file_names if f.endswith(".npy") and f != "TIMESTAMPS.npy"]
        if k>=10:
            file_names = natsorted(os.listdir(abs_path+f"/The cloud dataset/{year}M{k}"))
            files += [os.path.join(f"{path_in}/{year}M{k}", f) for f in file_names if f.endswith(".npy") and f != "TIMESTAMPS.npy"]
            
    len_files = len(files)
    list_new_array, list_new_array_pca, list_new_array_border, list_new_array_border_pca = [], [], [], []
    for t in tqdm(range(0, len_files, 8)):
        new_array, new_array_pca, new_array_border, new_array_border_pca = merged_arrays(files[t], num_labels, slices, k_main_components)
        
        list_new_array.append(xr.DataArray(new_array, dims=('label', 'x', 'y')))
        list_new_array_pca.append(xr.DataArray(new_array_pca, dims=('label', 'x', 'y')))
        list_new_array_border.append(xr.DataArray(new_array_border, dims=('label', 'x', 'y')))
        list_new_array_border_pca.append(xr.DataArray(new_array_border_pca, dims=('label', 'x', 'y')))
    
    len_dataset = len(list_new_array)
    
    d1 = xr.DataArray(list_new_array, 
                 dims=('time', 'label', 'x', 'y'), 
                 coords={'time': range(len_dataset)},
                 name='data')
    d1.to_netcdf(f"{path_out}/new_array.nc", 
                            mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': compression}})
    
    d2 = xr.DataArray(list_new_array_pca, 
                 dims=('time', 'label', 'x', 'y'), 
                 coords={'time': range(len_dataset)},
                 name='data')
    d2.to_netcdf(f"{path_out}/new_array_pca.nc", 
                            mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': compression}})
    
    d3 = xr.DataArray(list_new_array_border, 
                 dims=('time', 'label', 'x', 'y'), 
                 coords={'time': range(len_dataset)},
                 name='data')
    d3.to_netcdf(f"{path_out}/new_array_border.nc", 
                            mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': compression}})
    
    d4 = xr.DataArray(list_new_array_border_pca, 
                 dims=('time', 'label', 'x', 'y'), 
                 coords={'time': range(len_dataset)},
                 name='data')
    d4.to_netcdf(f"{path_out}/new_array_border_pca.nc", 
                            mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': compression}})
    
    return new_array, new_array_pca, new_array_border, new_array_border_pca

# %%
path_in = abs_path+"/The cloud dataset"
path_out = abs_path+"/P3_Training"
new_array, new_array_pca, new_array_border, new_array_border_pca = save_dataset(path_in, path_out, year="2017", num_labels=4, slices=6)

# %%
path_in = abs_path+"/The cloud dataset"
path_out = abs_path+"/P3_Testing"

new_array, new_array_pca, new_array_border, new_array_border_pca = save_dataset(path_in, path_out, year="2018", num_labels=4, slices=6)

# %%
files = []
for k in range(1, 13):
    if k <=9:
        file_names = natsorted(os.listdir(abs_path+f"/The cloud dataset/2017M0{k}"))
        files += [os.path.join(f"{path_in}/2017M0{k}", f) for f in file_names if f.endswith(".npy") and f != "TIMESTAMPS.npy"]
    if k>=10:
        file_names = natsorted(os.listdir(abs_path+f"/The cloud dataset/2017M{k}"))
        files += [os.path.join(f"{path_in}/2017M{k}", f) for f in file_names if f.endswith(".npy") and f != "TIMESTAMPS.npy"]

# %%

d1 = xr.open_dataset(abs_path+"/Files/new_array.nc")['data'].values
d2 = xr.open_dataset(abs_path+"/Files/new_array_pca_fnormalized.nc")['data'].values
d3 = xr.open_dataset(abs_path+"/Files/new_array_border.nc")['data'].values
d4 = xr.open_dataset(abs_path+"/Files/new_array_border_pca_fnormalized.nc")['data'].values

# %%
plt.subplots(4, 4)
j=4000
for k in range(4):
    plt.subplot(4, 4, k+1)
    plt.imshow(d1[j,k,:,:], cmap='gray')
    plt.subplot(4, 4, k+5)
    plt.imshow(d2[j,k,:,:], cmap='gray')
    plt.subplot(4, 4, k+9)
    plt.imshow(d3[j,k,:,:], cmap='gray')
    plt.subplot(4, 4, k+13)
    plt.imshow(d4[j,k,:,:], cmap='gray')
    
# %%
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig(abs_path+"/12_new_4x_sliced_arrays.png")
plt.close('all')

# %%

def data_toarray_selection(path_in, variable, num_labels=4):
        
    dataset = xr.open_dataset(path_in)
    darray = dataset[variable].values
    
    len_time, num_labels, m, n = darray.shape

    y = np.zeros((len_time*num_labels))
    X = np.zeros((len_time*num_labels, m*n))
        
    for t in tqdm(range(len_time*num_labels)):
        X[t, :] = darray[t//num_labels, t%num_labels, :, :].flatten()
        y[t] = t%num_labels
    
    return X, y


def run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=4, itterations=100):
    
    X, y = data_toarray_selection(files, variable, num_labels)
    print("Training dataset of size : ", X.shape)

    Theta1, Theta2 = ML.nnOneLayer(X, y, hidden_layer_size, num_labels, itterations)
        
    return Theta1, Theta2


def test_datasetNN(files, Theta1, Theta2, variable="data", sample_size=100, num_labels=13):
    
    X, y = data_toarray_selection(files, variable, num_labels)
    p_arrays = ML.predOneLayer(X, Theta1, Theta2)
    
    len_sample, len_data = X.shape
    print("Testing dataset of size : ", X.shape)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, len_sample, num_labels):
            if p_arrays[t] == k:
                success_of_label_k+=1
        success_of_label_k = trunc(success_of_label_k/(len_sample/num_labels)*10000)
        print("Success of label", k, "is : ", success_of_label_k/100, "%")

    return p_arrays

# %% Computation of images NN
files = abs_path+"/P3_Training/new_array_pca.nc"

Theta1, Theta2 = run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=4, itterations=100)

np.save(abs_path+"/P3_Training/trained_theta_NN_new_array_pca.npy", np.array([Theta1, Theta2], dtype=object))

# %% Testing of images NN, training dataset
files = abs_path+"/P3_Training/new_array_pca.nc"
Theta1, Theta2 = np.load(abs_path+"/P3_Training/trained_theta_NN_new_array_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1, Theta2, variable="data", num_labels=4)

# %% Testing of images NN, testing dataset
files = abs_path+"/P3_Testing/new_array_pca.nc"
Theta1, Theta2 = np.load(abs_path+"/P3_Training/trained_theta_NN_new_array_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1, Theta2, variable="data", num_labels=4)

# %% Computation of border NN
files = abs_path+"/P3_Training/new_array_border_pca.nc"

Theta1_, Theta2_ = run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=13, itterations=100)

np.save(abs_path+"/P3_Training/trained_theta_NN_new_array_border_pca.npy", np.array([Theta1_, Theta2_], dtype=object))

# %% Testing of border NN, training dataset
files = abs_path+"/P3_Training/new_array_border_pca.nc"
Theta1_, Theta2_ = np.load(abs_path+"/P3_Training/trained_theta_NN_new_array_border_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1_, Theta2_, variable="data", num_labels=4)

# %% Testing of border NN, testing dataset
files = abs_path+"/P3_Testing/new_array_border_pca.nc"
Theta1_, Theta2_ = np.load(abs_path+"/P3_Training/trained_theta_NN_new_array_border_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1_, Theta2_, variable="data", num_labels=4)

# %%
#================ END OF PART III ================#






















