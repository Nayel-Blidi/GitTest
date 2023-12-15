# %%
#================ PART IV ================#

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import sys

from tqdm import tqdm
from scipy.signal import convolve2d
from natsort import natsorted
from math import trunc

import ML

"""
Specify the absolute path of the 'Project' folder location :
"""
abs_path = "D:/Machine Learning/Project"

#The cloud dataset, 6go, np arrays 30+go
#https://www.kaggle.com/datasets/christianlillelund/the-cloudcast-dataset?resource=download
#https://vision.eng.au.dk/cloudcast-dataset/

#The cloud dataset : dataset of suboptimal npy arrays to dataarray

#Data is structured in 24 files, for every month, named {year}M{month}
#Every npy is the labelized cloud category, at t={num_array}*15 minutes
#The data is studied above europe, centered on France, and covers some parts of the Maghreb
#More especially between the long and lats : cf GEO.npz

TestCloud = xr.open_dataset(abs_path+"/small_cloud/TestCloud.nc")["__xarray_dataarray_variable__"]
TrainingCloud = xr.open_dataset(abs_path+"/small_cloud/TrainCloud.nc")["__xarray_dataarray_variable__"]

array_info = TrainingCloud.values[:,:,0]
array_info[array_info==255] = 4

m, n, len_time = TrainingCloud.shape

# %%
"""
Categories are initially as followed :
    1-3 and 255 represent land caracteristics, and are thus considered zeros
    4-8 are very low, low, mid, high and very high clouds
    9 are fractionnal clouds
    10-13 are semitransparent high clouds
Categories will be merged as :
    0 land caracteristics
    1-5 very low, low, mid, high and very high clouds
    6 fractionnal and semi transparent high clouds
"""

plt.hist(TrainingCloud.values[:,:,0].ravel(), bins=12, range=(1, 13))
plt.title("Pixels distribution by category before merging")
plt.savefig(abs_path+"/15_pixels_histogram_before_merging.png")

# %%

def label_merging(dataset):
    data = dataset.copy()
    m, n, len_time = data.shape
    
    data_hist = data.values[:,:,0].ravel()
    for t in tqdm(range(len_time)):
        array = data[:,:,t].values
        #No cloud
        array[array==255] = 0
        array[array<=3] = 0
        #Low clouds
        array[array==4] = 1
        array[array==5] = 1
        #Mid clouds
        array[array==6] = 2
        #High clouds
        array[array>=7] = 3
        data[:,:,t] = array
            
    plt.hist(data.values.ravel(), bins=4, range=(0, 4))
    
    return data

# %%
path_out = abs_path+"/small_cloud/TrainCloud_merged.nc"
data_training = label_merging(TrainingCloud)
data_training.to_netcdf(path_out)

array = data_training[:,:,0]
plt.subplot(1,2,1)
plt.imshow(array, cmap='cool')
plt.subplot(1,2,2)
plt.hist(array.values.flatten(), bins=4, range=(0, 4))
plt.suptitle("Pixels count distribution, eg1")
plt.savefig(abs_path+"/16_pixels_categories_distribution_histogram.png")

# %%
path_out = abs_path+"/small_cloud/TestCloud_merged.nc"
data_test = label_merging(TestCloud)
data_test.to_netcdf(path_out)

array = data_test[:,:,1000]
plt.subplot(1,2,1)
plt.imshow(array, cmap='cool')
plt.subplot(1,2,2)
plt.hist(array.values.flatten(), bins=4, range=(0, 4))
plt.suptitle("Pixels count distribution, eg2")
plt.savefig(abs_path+"/17_pixels_categories_distribution_histogram.png")

# %%
#================ PCA new_dataset : Small Cloud ================#

# %%
def pca_array(array, k_main_components, normalization=True):
    """
    Module fonction.
    Runs PCA algorithm on an input array and keep the first k_main_components.
    Input array can be normalized or not.
    """
    array_pca, pca_info = ML.pca(array, k_main_components, normalization)
    return array_pca, pca_info


def label_merging_pca(array, num_labels=4, k_main_components=30):
        
    m, n = array.shape
    
    kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]])
    
    new_array = np.zeros((num_labels, m, n))
    new_array_pca = np.zeros((num_labels, m, k_main_components))
    for k in range(num_labels):
        array_label = np.zeros((m,n))
        array_label[array==k] = 1
        array_label[array!=k] = 0

        new_array[k, :, :] = array_label
        new_array_pca[k,:,:] = pca_array(array_label, k_main_components)[0]
    
    new_array_border = np.zeros((num_labels, m, n))
    new_array_border_pca = np.zeros((num_labels, m, k_main_components))
    for k in range(num_labels):
        new_array_border[k,:,:] = convolve2d(new_array[k,:,:], kernel, mode='same', boundary='symm')
        new_array_border_pca[k,:,:] = pca_array(new_array_border[k,:,:], k_main_components)[0]
        
    return new_array, new_array_pca, new_array_border, new_array_border_pca


def save_dataset(path_in, path_out, variable="__xarray_dataarray_variable__", num_labels=4, k_main_components=30, compression=False):
    """
    Save all the arrays in path_in folder into path_out folder, as merged datasets alongside dimension time.
    Calls the function merged_arrays
    """
    
    dataset = xr.open_dataset(path_in)[variable]
    m, n, len_time = dataset.shape

    list_new_array, list_new_array_pca, list_new_array_border, list_new_array_border_pca = [], [], [], []
    for t in tqdm(range(0, len_time, 8)):
        new_array, new_array_pca, new_array_border, new_array_border_pca = label_merging_pca(dataset[:,:,t], num_labels, k_main_components)
        
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
path_in = abs_path+"/small_cloud/TrainingCloud.nc"
path_out = abs_path+"/small_cloud/k10"

new_array, new_array_pca, new_array_border, new_array_border_pca = save_dataset(path_in, path_out, 
                                                                                num_labels=4, k_main_components=10, compression=True)

# %%
path_in = abs_path+"/small_cloud/TestCloud.nc"
path_out = abs_path+"/small_cloud/Testing_k30"

test_array, test_array_pca, test_array_border, test_array_border_pca = save_dataset(path_in, path_out, 
                                                                                    num_labels=4, k_main_components=30, compression=True)

# %%
d1 = xr.open_dataset(abs_path+"/small_cloud/new_array.nc")['data'].values
d2 = xr.open_dataset(abs_path+"/small_cloud/new_array_pca.nc")['data'].values
d3 = xr.open_dataset(abs_path+"/small_cloud/new_array_border.nc")['data'].values
d4 = xr.open_dataset(abs_path+"/small_cloud/new_array_border_pca.nc")['data'].values

# %%
for k in range(4):
    plt.subplot(2,2,k+1)
    plt.imshow(d1[0,k,:,:], cmap='cool')

# %%
#================ Neural Network - 2 ================#

# %%

def data_toarray_normalize_selection(path_in, variable, num_labels=4):
        
    dataset = xr.open_dataset(path_in)
    darray = dataset[variable].values
    
    len_time, num_labels, m, n = darray.shape

    y = np.zeros((len_time*num_labels))
    X = np.zeros((len_time*num_labels, m*n))
        
    for t in tqdm(range(len_time*num_labels)):
        X[t, :] = darray[t//num_labels, t%num_labels, :, :].flatten()
        y[t] = t%num_labels
        
    return X, y


def run_datasetNN(path_in, variable="data", hidden_layer_size=32, num_labels=4, itterations=100):
    
    print("Loading the normalized X and y to prevent overflows :")
    X, y = data_toarray_normalize_selection(path_in, variable, num_labels)
    print("Training dataset of size : ", X.shape)

    Theta1, Theta2 = ML.nnOneLayer(X, y, hidden_layer_size, num_labels, itterations)
        
    return Theta1, Theta2, X, y


def test_datasetNN(path_in, Theta1, Theta2, variable="data", sample_size=100, num_labels=13):
    
    print("Loading and normalizing X_test and y_test to match the training examples :")
    X, y = data_toarray_normalize_selection(path_in, variable, num_labels)
    
    len_sample, len_data = X.shape
    print("Testing dataset of size : ", X.shape)
    
    p_arrays = ML.predOneLayer(X, Theta1, Theta2)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, len_sample, num_labels):
            if p_arrays[t] == k:
                success_of_label_k+=1
        success_of_label_k = trunc(success_of_label_k/(len_sample/num_labels)*10000)
        print("Success of label", k, "is : ", success_of_label_k/100, "%")

    return p_arrays

# %% Computation of images NN
files = abs_path+"/small_cloud/k30/new_array_pca.nc"

Theta1, Theta2, X, y = run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=4, itterations=100)

np.save(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_pca.npy", np.array([Theta1, Theta2], dtype=object))

# %% Testing of images NN, training dataset
files = abs_path+"/small_cloud/k30/new_array_pca.nc"
Theta1, Theta2 = np.load(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1, Theta2, variable="data", num_labels=4)

# %% Testing of images NN, testing dataset
files = abs_path+"/small_cloud/Testing_k30/new_array_pca.nc"
Theta1, Theta2 = np.load(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1, Theta2, variable="data", num_labels=4)

# %% Computation of border NN
files = abs_path+"/small_cloud/k30/new_array_border_pca.nc"

Theta1_, Theta2_, X, y = run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=4, itterations=100)

np.save(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_border_pca.npy", np.array([Theta1_, Theta2_], dtype=object))

# %% Testing of border NN, training dataset
files = abs_path+"/small_cloud/k30/new_array_border_pca.nc"
Theta1_, Theta2_ = np.load(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_border_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1_, Theta2_, variable="data", num_labels=4)

# %% Testing of border NN, testing dataset
files = abs_path+"/small_cloud/Testing_k30/new_array_border_pca.nc"
Theta1_, Theta2_ = np.load(abs_path+"/small_cloud/k30/trained_theta_NN_new_array_border_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1_, Theta2_, variable="data", num_labels=4)

# %%

def imageAndBorders(path_in, path_out):
    
    ds1 = xr.open_dataset(f"{path_in}/new_array_pca.nc")
    ds2 = xr.open_dataset(f"{path_in}/new_array_border_pca.nc")

    # Concatenate the datasets along array axis=1
    dataset_combined = xr.concat([ds1, ds2], dim='y')
    
    dataset_combined.to_netcdf(f"{path_out}/new_combined_pca.nc")
    
    return dataset_combined

# %% Concatenation of array and border datasets
path_in = abs_path+"/small_cloud"
path_out = abs_path+"/small_cloud"

dataset_combined = imageAndBorders(path_in, path_out)
darray_combined = dataset_combined["data"].values

# %% Computation of array&border NN
files = abs_path+"/small_cloud/new_combined_pca.nc"

Theta1_, Theta2_, X, y = run_datasetNN(files, variable="data", hidden_layer_size=32, num_labels=4, itterations=100)

np.save(abs_path+"/small_cloud/trained_theta_NN_new_combined_pca.npy", np.array([Theta1_, Theta2_], dtype=object))

# %% Testing of array&border NN, training dataset
files = abs_path+"/small_cloud/new_combined_pca.nc"
Theta1_, Theta2_ = np.load(abs_path+"/small_cloud/trained_theta_NN_new_combined_pca.npy", allow_pickle=True)

p_arrays = test_datasetNN(files, Theta1_, Theta2_, variable="data", num_labels=4)

# %%
#================ OneVsAll - 2 ================#

# %%

def OneVsAll_dataset(path_in, variable, lambda_=1, num_labels=4, itterations=50):
    
    print("Loading of X and y :")
    X, y = data_toarray_normalize_selection(path_in, variable, num_labels)
    
    len_sample, len_data = X.shape
    print("Testing dataset of size : ", X.shape)
    
    trained_theta = ML.oneVsAll(X, y, num_labels, lambda_, tol=1e-3)[1]
    
    return trained_theta, X, y


def OneVsAll_testDataset_pca(path_in, trained_theta, variable, sample_size=30, num_labels=13):
    
    X, y = data_toarray_normalize_selection(path_in, variable, num_labels)
    
    len_time, m = X.shape
    
    p = ML.predictOneVsAll(trained_theta[:,1:], X)
    
    for k in range(num_labels):
        success_of_label_k = 0
        for t in range(k, len_time, num_labels):
            if p[t] == k:
                success_of_label_k+=1
        success_of_label_k = success_of_label_k / (len_time/num_labels)
        print("Success of label", k, "is : ", success_of_label_k*100, "%")
    return p

# %%
"""
Due to the computation time being very steep (30sec/itt for k=10, 4min/itt for k=20, 25min/itt for k=30), calculation 
can't be runned for k>=30, thus forcing the size of array&border combined dataset to be capped at 2*(k=10) = 20.
""" 
# %% Computation of OneVsAll algorithm, image
path_in = abs_path+"/small_cloud/k30/new_array_pca.nc"

trained_theta, X, y = OneVsAll_dataset(path_in, variable="data", lambda_=1, num_labels=4, itterations=50)

np.save(abs_path+"/small_cloud/trained_theta_OneVsAll_new_array_pca.npy", trained_theta)

# %% Testing of OneVsAll algorithm, image trained and test dataset
path_in = abs_path+"/small_cloud/k30/new_array_pca.nc"
trained_theta = np.load(abs_path+"/small_cloud/k30/trained_theta_OneVsAll_new_array_pca.npy")

p_arrays = OneVsAll_testDataset_pca(path_in, trained_theta, variable ="data", num_labels=4)

path_in = abs_path+"/small_cloud/Testing_k30/new_array_pca.nc"

p_arrays_test = OneVsAll_testDataset_pca(path_in, trained_theta, variable ="data", num_labels=4)

# %% Computation of OneVsAll algorithm, border
path_in = abs_path+"/small_cloud/k30/new_array_border_pca.nc"

trained_theta, X, y = OneVsAll_dataset(path_in, variable="data", lambda_=1, num_labels=4, itterations=50)

np.save(abs_path+"/small_cloud/k30/trained_theta_OneVsAll_new_array_border_pca.npy", trained_theta)

# %% Testing of OneVsAll algorithm, border trained and test dataset
path_in = abs_path+"/small_cloud/k30/new_array_border_pca.nc"
trained_theta = np.load(abs_path+"/small_cloud/k30/trained_theta_OneVsAll_new_array_border_pca.npy")

p_arrays = OneVsAll_testDataset_pca(path_in, trained_theta, variable ="data", num_labels=4)

path_in = abs_path+"/small_cloud/Testing_k30/new_array_border_pca.nc"

p_arrays_test = OneVsAll_testDataset_pca(path_in, trained_theta, variable ="data", num_labels=4)

# %% Computation of OneVsAll algorithm, combined dataset
path_in = abs_path+"/small_cloud/k10/new_combined_pca.nc"

trained_theta, X, y = OneVsAll_dataset(path_in, variable="data", lambda_=1, num_labels=4, itterations=50)

np.save(abs_path+"/small_cloud/k10/trained_theta_OneVsAll_new_combined_pca.npy", trained_theta)

# %% Testing of OneVsAll algorithm, combined dataset
path_in = abs_path+"/small_cloud/k10/new_combined_pca.nc"
trained_theta = np.load(abs_path+"/small_cloud/k10/trained_theta_OneVsAll_new_combined_pca.npy")

p_arrays = OneVsAll_testDataset_pca(path_in, trained_theta, variable ="data", num_labels=4)

# %%
#================ What if it worked perfectly ? ================#

# %%

def kmeans(image_array, num_clusters, itterations=1000):
    # Reshape the image array to a 2D array of pixels
    pixels = image_array
    
    # Randomly initialize cluster centers
    centers = pixels[np.random.choice(pixels.shape[0], num_clusters, replace=False)]
    
    # Iterate until convergence
    for k in tqdm(range(itterations)):
        # Assign pixels to the nearest cluster center
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update cluster centers
        new_centers = np.array([pixels[labels == i].mean(axis=0) for i in range(num_clusters)])
        
        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        
        centers = new_centers
    
    # Assign pixel values to the closest cluster center
    new_pixels = centers[labels]
    
    # Reshape the pixel values to match the original image shape
    new_image = new_pixels.reshape(image_array.shape)
    
    return new_image

num_clusters = 50
image_array = xr.open_dataset(abs_path+"/small_cloud/TrainCloud_merged.nc")["__xarray_dataarray_variable__"][:,:,0].values

plt.subplot(1,2,1)
plt.imshow(image_array)
array = kmeans(image_array, num_clusters)
plt.subplot(1,2,2)
plt.imshow(array)









