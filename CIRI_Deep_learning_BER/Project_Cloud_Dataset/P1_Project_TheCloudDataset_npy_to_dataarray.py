# %%
#================ PART I ================#

# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os

from tqdm import tqdm
from scipy.signal import convolve2d

import ML

#The cloud dataset, 6go, np arrays 30+go
#https://www.kaggle.com/datasets/christianlillelund/the-cloudcast-dataset?resource=download

#The cloud dataset : dataset of suboptimal npy arrays to dataarray

#Data is structured in 24 files, for every month, named {year}M{month}
#Every npy is the labelized cloud category, at t={num_array}*15 minutes
#The data is studied above europe, centered on France, and covers some parts of the Maghreb
#More especially between the long and lats : cf GEO.npz

"""
Specify the absolute path of the 'Project' folder location :
"""
abs_path = "D:/Machine Learning/Project"

data_array_0 = np.load(abs_path+"/The cloud dataset/2017M01/0.npy")
num_labels = np.max(data_array_0) - np.min(data_array_0) + 1
plt.imshow(data_array_0, cmap='cool')

lats = np.load(abs_path+"/The cloud dataset/2017M01/GEO.npz")['lats']
lons = np.load(abs_path+"/The cloud dataset/2017M01/GEO.npz")['lons']

# %%

def compute_mask_and_border(array, num_labels):
    
    m, n = array.shape
    kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])

    #Separation of the array masks by labels values
    list_array_sampled = []
    for k in range(1, num_labels+1):
        array_sampled = array.copy()
        array_sampled[array_sampled != k] = 0
        list_array_sampled.append((array_sampled.copy()//k).astype('int8'))

    #Convolution of the contours of every separated labels
    list_array_sampled_training_border = []
    for k in range(num_labels):
        image = list_array_sampled[k]
        border_image = convolve2d(image, kernel, mode='same', boundary='symm')
        border_image[border_image < 0] = 0
        border_image[border_image > 1] = 1
        #border_image = (border_image - np.min(border_image)) / (np.max(border_image) - np.min(border_image))
        list_array_sampled_training_border.append(border_image.astype('int8'))

    return list_array_sampled, list_array_sampled_training_border


def plot_mask_and_border(list_array_sampled, list_array_sampled_training_border, K=-1):

    fig, axs = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))     
    fig.subplots_adjust(hspace=0.6)
    plt.suptitle(f"Array {K}, separated labelized clouds and its borders")
    for k in tqdm(range(0, num_labels-1)):
        plt.subplot(4, 6, k+1)
        plt.imshow(list_array_sampled[k], cmap='cool')   
        plt.title(f"Separated label {k+1}")
        plt.subplot(4, 6, num_labels+k)
        plt.imshow(list_array_sampled_training_border[k], cmap='cool')    
        plt.title(f"Border label {k+1}")
        
    plt.savefig(abs_path+f"/Machine Learning/Project/Separated_array_{K}.png")
    return None

# %%
list_array_sampled, list_array_sampled_training_border = compute_mask_and_border(data_array_0, num_labels)
plot_mask_and_border(list_array_sampled, list_array_sampled_training_border, 0)

# %%%
"""
Now this is done for one array, let's use this function to edit all the arrays of a folder.

We'll then save all of these arrays in a more optimal extension, so all the data
(new_size = original_size * 26)
(new_size = (original_size=768²) * (num_labels=13) * (labels&borders=2)) 
can be loaded and stored more efficiently than as npy arrays
    
With all the data ready for learning, we'll start the neural network sorting processing in a dedicated file,
so all of the data mining calculation doesn't have to be calculated again.
"""
# %%

def generate_data_array(list_array, num_labels=13):
    
    data_array = xr.DataArray(np.stack(list_array), 
                              dims=('label', 'x', 'y'), 
                              coords={'label': range(num_labels)},
                              name='data')
    return data_array


def save_data_array(path_in, path_out_labels, path_out_borders, name_darray, 
                    batch_start, batch_end, batch_size=96, batch_info="", num_labels=13):
    
    try:
    #Some month samples are sometimes slightly smaller than a full month, so in this case,
    #we've choosed to don't store this data.
    #Learning data will be marginally smaller, but for every instance, all the 26 different categories 
    #of the datasets will have the same shape, which is more valuable.
            
        #Dimension informations
        timestamps = np.load(f"{path_in}{name_darray}/TIMESTAMPS.npy") 
        t = len(timestamps)
    
        #Initialisation of the dataaary with matrix 0
        array = np.load(f"{path_in}{name_darray}/{batch_start}.npy")
        
        #Creation of 2 lists for matrix k :
            #list_array_separated is the list of images with 1 label selected each
            #list_array_separated_border is the list if the borders of each image
        list_array_separated, list_array_separated_border = compute_mask_and_border(array, 
                                                                                    num_labels)
        
        #Conversion of both lists to data array, where label is the prime direction
        concatenated_darray_separated = generate_data_array(list_array_separated)
        concatenated_darray_separated_border = generate_data_array(list_array_separated_border)
        
        #The data arrays are concatenated for the whole folder
        for k in tqdm(range(batch_start+1, batch_end)):
            #Loading of matrix k
            array = np.load(f"{path_in}{name_darray}/{k}.npy")
            
            #Creation of 2 lists for matrix k :
                #list_array_separated is the list of images with 1 label selected each
                #list_array_separated_border is the list if the borders of each image
            list_array_separated, list_array_separated_border = compute_mask_and_border(array, 
                                                                                        num_labels)
            
            #Conversion of both lists to data array, where label=k is the prime direction
            data_array_separated = generate_data_array(list_array_separated, num_labels)
            data_array_separated_border = generate_data_array(list_array_separated_border, num_labels)
            
            #Concatenation
            concatenated_darray_separated = xr.concat([concatenated_darray_separated, 
                                                       data_array_separated], dim='time')
            concatenated_darray_separated_border = xr.concat([concatenated_darray_separated_border, 
                                                       data_array_separated_border], dim='time')

        #Saving of the new dataarrays
        concatenated_darray_separated.to_netcdf(f"{path_out_labels}/{name_darray}_concatenated_darray_separated{batch_info}.nc", 
                                                mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': True}})
        concatenated_darray_separated_border.to_netcdf(f"{path_out_borders}/{name_darray}_concatenated_darray_separated_borders{batch_info}.nc", 
                                                       mode='w', format='netCDF4', engine='netcdf4', encoding={'data': {'zlib': True}})
    
    #Let's just not add anything to our
    except:
        return None, None
        
    return concatenated_darray_separated, concatenated_darray_separated_border

# %%
path_in = abs_path+"/The cloud dataset/"
path_out_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_out_borders = abs_path+"/TheCloudDataset_labelized_borders_train"
name_darray = "2017M01"

dataarray_labels, dataarray_borders = save_data_array(path_in, path_out_labels, path_out_borders, name_darray,
                                                     batch_start=0, batch_end=96, num_labels=13)

# %%

#Due to performance issue, dataarray had to be batched
def batched_save_data_array(path_in, path_in_labels, path_out_borders, name_darray, 
                            batch_size=96, num_labels=13):
    
    #Dimension informations
    timestamps = np.load(f"{path_in}{name_darray}/TIMESTAMPS.npy") 
    t = len(timestamps)
    
    #Batch size is 1 day, over 31 days in the case of January, 28 for February... 
    for k in tqdm(range(0, t-batch_size, batch_size)):
        #Every dataarray is by default a batch of (1 day = 1*24*4 = 96), 15min time intervals
        batch_info = f"_{k//batch_size}"
        batch_start = k
        batch_end = k+batch_size
        dataarray_labels, dataarray_borders = save_data_array(path_in, path_out_labels, path_out_borders, name_darray,
                                                             batch_start, batch_end, batch_size, batch_info, 13)
    return None

# %%
path_in = abs_path+"/The cloud dataset/"
path_out_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_out_borders = abs_path+"/TheCloudDataset_labelized_borders_train"
name_darray = "2017M01"

#Computation takes roughly 20min for a month
batched_save_data_array(path_in, path_out_labels, path_out_borders, name_darray,
                        batch_size=96, num_labels=13)  

# %%
path_in = abs_path+"/The cloud dataset/"
path_out_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_out_borders = abs_path+"/TheCloudDataset_labelized_borders_train"

#Dirty loop over the whole year 2017
#REFAIRE M02 0-4
for k in range(7, 11):
    if k <= 8:
        name_darray = f"2017M0{k+1}"
        batched_save_data_array(path_in, path_out_labels, path_out_borders, name_darray,
                                batch_size=96, num_labels=13)  
    elif k >= 9:
        name_darray = f"2017M{k+1}"
        batched_save_data_array(path_in, path_out_labels, path_out_borders, name_darray,
                                batch_size=96, num_labels=13)  

# %%
"""
Data is now packed in 12 forlders over the year 2017, as netcdf arrays

To sumarize the compression, let's take the example of one day in January 2017:
    One day in npy format was 24*4*577kb = 55392kb
    Now, for one day, there are 26 time more values (13 labels, 2 categories), and
    the file size for one array is 18577 + 20456 = 39033kb
    42% larger, yet storing 26 times less data
    
It is now possible to go on the next step, and start the learning of the cloud patterns
(might regroup all the data in 2 large netcdf dataarrays)

"""
# %%

path_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_borders = abs_path+"/TheCloudDataset_labelized_borders_train"

def merge_data_array(path_labels = abs_path+"/TheCloudDataset_labelized_arrays_train",
                     path_borders = abs_path+"/TheCloudDataset_labelized_borders_train"):
    
    #Merging of all the daily file into one full year dataset
    files_darray_labels = [os.path.join(path_labels, f) for f in os.listdir(path_labels) if f.endswith(".nc")]
    files_darray_borders = [os.path.join(path_borders, f) for f in os.listdir(path_borders) if f.endswith(".nc")]

    #merged_darray_labels = xr.open_mfdataset(files_darray_labels, combine='nested', concat_dim="time")
    #merged_darray_borders = xr.open_mfdataset(files_darray_borders, combine='nested', concat_dim="time")
    
    merged_darray_labels = xr.open_dataset(files_darray_labels[0])
    for k in tqdm(range(1, len(files_darray_labels)-200)):
        array = xr.open_dataset(files_darray_labels[k])
        merged_darray_labels = xr.concat([merged_darray_labels, array], dim="time")
    merged_darray_labels.to_netcdf(f"{path_labels}/2017M00_merged_darray_labels.nc")

    merged_darray_borders = xr.open_dataset(files_darray_labels[0])
    for k in tqdm(range(1, len(files_darray_borders)-238)):
        array = xr.open_dataset(files_darray_borders[k])
        merged_darray_borders = xr.concat([merged_darray_borders, array], dim="time")
    merged_darray_borders.to_netcdf(f"{path_borders}/2017M00_merged_darray_borders.nc")

    return merged_darray_labels, merged_darray_borders

merged_darray_labels, merged_darray_borders = merge_data_array(path_labels, path_borders)

# %%

path_labels = abs_path+"/TheCloudDataset_labelized_arrays_train"
path_borders = abs_path+"/TheCloudDataset_labelized_borders_train"

def merge_data_array_bis(path_labels = abs_path+"/TheCloudDataset_labelized_arrays_train",
                             path_borders = abs_path+"/TheCloudDataset_labelized_borders_train"):
    
    #Merging of all the daily file into one full year 
    files_darray_labels = [os.path.join(path_labels, f) for f in os.listdir(path_labels) if f.endswith(".nc")]
    files_darray_borders = [os.path.join(path_borders, f) for f in os.listdir(path_borders) if f.endswith(".nc")]
    
    # open the files using dask
    merged_darray_labels = xr.open_mfdataset(files_darray_labels, combine='nested', concat_dim='time', engine='netcdf4', chunks={'time': 'auto'})
    merged_darray_borders = xr.open_mfdataset(files_darray_borders, combine='nested', concat_dim='time', engine='netcdf4', chunks={'time': 'auto'})

    # save the concatenated dataset to a new netCDF file
    merged_darray_labels.to_netcdf(f"{path_labels}/2017M00_merged_darray_labels.nc", mode='w')
    merged_darray_borders.to_netcdf(f"{path_borders}/2017M00_merged_darray_borders.nc", mode='w')

    return merged_darray_labels, merged_darray_borders

merged_darray_labels, merged_darray_borders = merge_data_array_bis(path_labels, path_borders)

# %%
#================ END OF PART I ================#

# %%











    

