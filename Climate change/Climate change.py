#https://berkeleyearth.org/data/
#https://climatedataguide.ucar.edu/climate-data/global-surface-temperatures-best-berkeley-earth-surface-temperatures
#https://climatedataguide.ucar.edu/climate-data/global-land-ocean-surface-temperature-data-hadcrut5
#https://www.nccs.nasa.gov/services/climate-data-services

#Bekerley earth is a summary of the following studies :
    #https://berkeleyearth.org/archive/source-files/

#Feuille de route :
    # 1) Collecte des données : site de la nasa
    # 2) Conversion des données, extraction depuis les fichiers netcdf, conversion non concluante
    # en dataset pandas et séries, puis en dictionnaire. Après réflexion, dataset aurait fonctionné
    # 3) Sauvegarde des sets de donnés en matrices hors dicitonnaire et sorties vidéos des plots
    # 4) Début du travail de data science :
        # a1) Interpolation des températures terrestes de surface dans le passé (ML)
        # a2) Interpolation des températures océaniques de surface dans le passé (ML)
        # b1) Entrainement à la prédiction de température terrestre par celle des océans
        # b2) Entrainement à la prédiction de température océanique par celle terrestre
        # c) Prédiction de l'évolution de température future, océanique, terrestre, des deux
        # d) Ajout de facteurs dans la prédiciton (ie l'apprentissage) :
            # - Précipitations
            # - Pression ?
            # - Vents ?
            # - ?

import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import sys
from tqdm import tqdm
import math
from scipy.stats import pearsonr

import ML

#"C:/Travail/IPSA/Aero4/Module électif/NCFiles/Bekerley_Land_Avg_Monthly.nc"
#"C:/Travail/IPSA/Aero4/Module électif/NCFiles/Bekerley_Land_Avg_EqualArea.nc"

#"C:/Users/BLIDI/Downloads/Bekerley_Land_Avg_Monthly.nc"
#"C:/Users/BLIDI/Downloads/Bekerley_Land_Avg_EqualArea.nc"
#"C:/Users/BLIDI/Downloads/Land_and_Ocean_Avg_Monthly.nc"

file_Berkeley_data_path = "C:/Users/BLIDI/Downloads/Land_and_Ocean_Avg_Monthly.nc"

data = xr.open_dataset(file_Berkeley_data_path)
print(data)

#Conversion en dictionnaire de dictionnaire l'array de arrays
dict_full = data.to_dict()

# Selection des pages du dico chapitre "Coordonnées"
longitude = dict_full['coords']['longitude']['data']
latitude = dict_full['coords']['latitude']['data']
l_time = dict_full['coords']['time']['data']

#t (mois) l'indice du temps, varie entre 3276 et ~2000 valeurs selon les bases de données.
#De fait, année_0 + t/12 = année_t
t = len(l_time)
#De fait, le fichier data fait 360*180*3276 valeurs, soit 212M

dict_temp = dict_full['data_vars']['temperature']['data']

#The full dictionnary is composed as followed:
    #coords : longitude, latitude, time
    #attrs: Conventions title history institution source_file source_history source_code_version comment
    #dims : time, latitude, longitude
    #data_vars :  :
        #land_mask
        #climatology
        #temperature
            #dims : time latitude longitude (tuple)
            #attrs : Air Surface Temperature Anomaly, surface_temperature_anomaly degree C, 
            #       17.81180370788261 (valid_max), -16.157561633762512 (valid_min)
        
keys1 = list(dict_full.keys())
keys2 = list(dict_full['data_vars'].keys())
keys3 = list(dict_full['data_vars']['temperature'].keys())

print(list(keys1), list(keys2), list(keys3))

#%% Test drawcoastlines()
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

#Test of the map at the moment t = 2000
test_data = dict_full['data_vars']['temperature']['data'][2000]
test_data = np.array(test_data)
test_data = test_data[::-1, ::1]

fig = plt.figure(figsize=(10, 6))

m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines(linewidth=0.3, color='black')

x, y = np.meshgrid(np.linspace(m.llcrnrlon, m.urcrnrlon, test_data.shape[1]), np.linspace(m.llcrnrlat, m.urcrnrlat, test_data.shape[0]))
xx, yy = m(x, y)

im = plt.imshow(test_data)
plt.pcolor(xx, yy, test_data)

year = math.trunc(l_time[2005])
date = year + round(((l_time[2005]-year) * 120)/100, 2)
plt.title(date)
plt.savefig("C:/Travail/IPSA/Aero4/Module électif/img_overlayed")
#matrix_test = im.get_array()
#plt.close()

#%%
#================OUTPUT VIDEO================#

#%% Save data plot with drawcoastlines() mask on top as a png
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

# /!\ Calculation time warning : takes roughly 30 minutes !

fig = plt.figure(figsize=(10, 6))
test_data = dict_full['data_vars']['temperature']['data'][0]
test_data = np.array(test_data)

m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
m.drawcoastlines(linewidth=0.3, color='black')

x, y = np.meshgrid(np.linspace(m.llcrnrlon, m.urcrnrlon, test_data.shape[1]), np.linspace(m.llcrnrlat, m.urcrnrlat, test_data.shape[0]))
xx, yy = m(x, y)

#Saving the dictionnary's data as numpy arrays in a folder, with a coastline mask
for k in tqdm(range(t)):
    fig = plt.figure(figsize=(10, 6))
    m.drawcoastlines(linewidth=0.3, color='black')
    im = plt.imshow(dict_temp[k])
    plt.pcolor(xx, yy, np.array(dict_temp[k]))
    plt.title(f"{l_time[k]}")
    plt.savefig(f"C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_and_Ocean_Avg_Monthly_Overlayed_PNG/img_overlayed_{k}")
    plt.close()

plt.close()

#%%
import os
import datetime
import imageio
from tqdm import tqdm

# Set the path to the folder containing the png files
path = "C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_and_Ocean_Avg_Monthly_Overlayed_PNG/"

# Get a list of all png files in the folder
files = [f for f in os.listdir(path) if f.endswith(".png")]

# Sort the files by date of creation
files = sorted(files, key=lambda x: os.path.getctime(os.path.join(path, x)))
print("Files sucesfully sorted by date of creation")

# Create a list of imageio Reader objects for each png file
images = [imageio.imread(os.path.join(path, f)) for f in files]

# Set the output file name and fps
output_file = "C:/Travail/IPSA/Aero4/Module électif/output.mp4"
fps = 30

# Use imageio to create the video
k=0
with imageio.get_writer(output_file, fps=fps) as writer:
    for image in tqdm(images):
        writer.append_data(image)


#%%
#Generate gif out of a folder of png
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import imageio
import os

png_folder = "C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_and_Ocean_Avg_Monthly_Overlayed_PNG"
gif_name = 'C:/Travail/IPSA/Aero4/Module électif/World_Land_and_Ocean_Temperature_Monthly_Avg.gif'

images = []
for file_name in os.listdir(png_folder):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_folder, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave(gif_name, images)

#%%
#Save data as a numpy array
from tqdm import tqdm

#Saving the dictionnary's data as numpy arrays in a folder
for k in tqdm(range(t)):
    matrix = np.array(dict_temp[k])
    np.save(f"C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_and_Ocean_Avg_Monthly/matrix_{k}.npy", matrix)

#%%
#Generate gif video out of a folder of matrixes

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# create the imshow plot
fig, ax = plt.subplots()
matrix = np.load("C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_Monthly_Avg_NaN/matrix_0.npy")
im = ax.imshow(matrix, animated=True)

def update_matrix(frame):
    #for k in range(1, t):
    matrix = np.load(f"C:/Travail/IPSA/Aero4/Module électif/Matrixes_Land_Monthly_Avg_NaN/matrix_{frame}.npy")
    matrix = matrix[::-1, ::1]    
    im.set_array(matrix)
    return [im]

# create the animation
ani = animation.FuncAnimation(fig, update_matrix, frames=10, blit=True)

# save the animation as a gif
ani.save('C:/Travail/IPSA/Aero4/Module électif/matrix_animation_Land_and_Ocean_2.gif', writer='imagemagick')

plt.close()
# display the animation
#plt.show()

#%%
#================GRAPHS & IMAGES================#

#%%
#plot_Global__Temperature returns :
    #mean_continent plots : the evolution of the average temperature on earth, month by month, divided
    #                       in 4 parts, relative to the average temperature before 1850 as graphs
    #mean_matrix : the average temperature evolution these 150 last years, on every point of the world
    #               Is is notably noticeable that the average temperature is roughly the same
    #               on land and closer to the equator, but the surface temperature greatly increased
    #               to the pole. 
    #               Thus, land average is higher in artic and antartic, whereas ocean temperatures are
    #               diminishing near by the shores, where the ice is smelting

def compute_Global_Temperature_Avg():
    lim_longitude = 160
    lim_latitude = 80
    m, n = 180, 360
    
    mean_continent = [[], [], [], []]
    for k in tqdm(range(t)):
        mean_continent[0].append(np.nanmean(np.array(dict_temp[k])[0:lim_latitude, 0:lim_longitude]))
        mean_continent[1].append(np.nanmean(np.array(dict_temp[k])[0:lim_latitude, lim_longitude:]))
        mean_continent[2].append(np.nanmean(np.array(dict_temp[k])[lim_latitude:, 0:lim_longitude]))
        mean_continent[3].append(np.nanmean(np.array(dict_temp[k])[lim_latitude:, lim_longitude:]))

    mean_matrix = np.array(np.ones((m, n)))
    for i in tqdm(range(m)):
        for j in range(n):
            mean_coords = np.zeros(t)      
            for k in range(t):
                mean_coords[k] = dict_temp[k][i][j]
            mean_matrix[i, j] = np.nanmean(mean_coords)
    
    return mean_continent, mean_matrix

mean_continent, mean_matrix = compute_Global_Temperature_Avg()

#%%
def plot_Global_Temperature_Avg(mean_continent, mean_matrix):
    lim_longitude = 160
    lim_latitude = 80

    #Graphs of the average variations by continents
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.6)
    fig.set_figheight(6)
    fig.set_figwidth(10)
    
    axs[0, 0].plot(l_time, mean_continent[0])
    axs[0, 0].set_title("Average temperature in North America")
    axs[0, 0].grid(True)
    axs[1, 0].plot(l_time, mean_continent[1])
    axs[1, 0].set_title("Average temperature in South America")
    axs[1, 0].grid(True)
    axs[0, 1].plot(l_time, mean_continent[2])
    axs[0, 1].set_title("Average temperature in Europe and East Asia")
    axs[0, 1].grid(True)
    axs[1, 1].plot(l_time, mean_continent[3])
    axs[1, 1].set_title("Average temperature in South East hemisphere")
    axs[1, 1].grid(True)
    plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/mean_4_continents_plots")
    
    #Image of the verage variation on earth
    fig, ax = plt.subplots()
    fig.figsize=(7, 9)

    m = Basemap(projection='mill', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
    m.drawcoastlines(linewidth=0.3, color='black')
    x, y = np.meshgrid(np.linspace(m.llcrnrlon, m.urcrnrlon, mean_matrix.shape[1]), np.linspace(m.llcrnrlat, m.urcrnrlat, mean_matrix.shape[0]))
    xx, yy = m(x, y)

    im = ax.imshow(mean_matrix[::-1, ::1])
    plt.pcolor(xx, yy, mean_matrix)
    colorbar = ax.figure.colorbar(im, ax=ax)
    colorbar.ax.text(3.5, 0.4, 'Variations (°C)', transform=colorbar.ax.transAxes, rotation=90)
    plt.title("Global average temperature variation\nbetween 1850 and nowaday")
    plt.axvline(lim_longitude, color='red', linewidth=0.6)
    plt.axhline(lim_latitude, color='red', linewidth=0.6)
    plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/mean_world_matrix")
    
    #plt.close('all')
    return None

plot_Global_Temperature_Avg(mean_continent, mean_matrix)

#%%
def compute_Hemisphere_Temperature_Avg():
    lim_longitude = 180
    lim_latitude = 90
    m, n = 180, 360
    year_start = 600
    
    mean_hemisphere = [[], []]
    for k in tqdm(range(year_start, t)):
        mean_hemisphere[0].append(np.nanmean(np.array(dict_temp[k])[0:lim_latitude, :]))
        mean_hemisphere[1].append(np.nanmean(np.array(dict_temp[k])[lim_latitude:, :]))

    mean_north_hemisphere_winter = []
    mean_north_hemisphere_summer = []
    mean_south_hemisphere_winter = []
    mean_south_hemisphere_summer = []
    for k in tqdm(range(12, t-year_start, 12)):
        north_hemisphere_winter = mean_hemisphere[0][k-1] + mean_hemisphere[0][k] + mean_hemisphere[0][k+2]
        mean_north_hemisphere_winter.append(north_hemisphere_winter)
        north_hemisphere_summer = mean_hemisphere[0][k+6] + mean_hemisphere[0][k+7] + mean_hemisphere[0][k+8]
        mean_north_hemisphere_summer.append(north_hemisphere_summer)
        
        south_hemishpere_winter = mean_hemisphere[1][k-1] + mean_hemisphere[1][k] + mean_hemisphere[1][k+2]
        mean_south_hemisphere_winter.append(south_hemishpere_winter)
        north_hemishpere_south = mean_hemisphere[1][k+6] + mean_hemisphere[1][k+7] + mean_hemisphere[1][k+8]
        mean_south_hemisphere_summer.append(north_hemishpere_south)
        
    return year_start, mean_hemisphere, mean_north_hemisphere_winter, mean_north_hemisphere_summer, mean_south_hemisphere_winter, mean_south_hemisphere_summer

year_start, mean_hemisphere, mean_north_hemisphere_winter, mean_north_hemisphere_summer, mean_south_hemisphere_winter, mean_south_hemisphere_summer = compute_Hemisphere_Temperature_Avg()

#%%
def plot_Hemisphere_Temperature_Avg(year_start, mean_north_hemisphere_winter, mean_north_hemisphere_summer, mean_south_hemisphere_winter, mean_south_hemisphere_summer):
    lim_longitude = 160
    lim_latitude = 80
    
    #Graphs of the average variations by hemisphere and seasons
    fig, axs = plt.subplots(4, 1)
    fig.subplots_adjust(hspace=0.4)
    fig.set_figheight(9)
    fig.set_figwidth(8)
    
    time_year = np.linspace(l_time[year_start], 2023, len(mean_north_hemisphere_summer))
    order = 3
    
    coeffs = np.polyfit(time_year, mean_north_hemisphere_winter, order)
    x_vals = time_year
    y_vals = np.polyval(coeffs, x_vals)
    
    coeffs_2 = np.polyfit(time_year, mean_north_hemisphere_summer, order)
    x_vals_2 = time_year
    y_vals_2 = np.polyval(coeffs_2, x_vals_2)
    
    coeffs_3 = np.polyfit(time_year, mean_south_hemisphere_winter, order)
    x_vals_3 = time_year
    y_vals_3 = np.polyval(coeffs_3, x_vals_3)
    
    coeffs_4 = np.polyfit(time_year, mean_south_hemisphere_summer, order)
    x_vals_4 = time_year
    y_vals_4 = np.polyval(coeffs_4, x_vals_4)
    
    plt.subplot(4, 1, 1)
    plt.plot(time_year, mean_north_hemisphere_winter)
    plt.title("Average winter temperature variation in North Hemisphere")
    plt.plot(x_vals, y_vals)
    plt.plot(x_vals_2, y_vals_2)
    corr_coeff_north_winter = pearsonr(mean_north_hemisphere_winter, y_vals)[0]
    textstr = f'Correlation coefficient: {corr_coeff_north_winter:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1900, 2, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    plt.grid()
    
    plt.subplot(4, 1, 2)
    plt.plot(time_year, mean_north_hemisphere_summer)
    plt.title("Average summer temperature variation in North Hemisphere")
    plt.plot(x_vals, y_vals)  
    plt.plot(x_vals_2, y_vals_2)
    corr_coeff_nprth_summer = pearsonr(mean_north_hemisphere_summer, y_vals_2)[0]
    textstr = f'Correlation coefficient: {corr_coeff_nprth_summer:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1900, 2, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(time_year, mean_south_hemisphere_winter)
    plt.title("Average winter temperature in South Hemisphere")
    plt.plot(x_vals_3, y_vals_3)
    plt.plot(x_vals_4, y_vals_4)
    corr_coeff_south_winter = pearsonr(mean_south_hemisphere_winter, y_vals_3)[0]
    textstr = f'Correlation coefficient: {corr_coeff_south_winter:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1900, 4, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    plt.grid()
    
    plt.subplot(4, 1, 4)
    plt.plot(time_year, mean_south_hemisphere_summer)
    plt.title("Average summer temperature in South Hemisphere")
    plt.plot(x_vals_3, y_vals_3)
    plt.plot(x_vals_4, y_vals_4)
    corr_coeff_south_summer = pearsonr(mean_south_hemisphere_summer, y_vals_4)[0]
    textstr = f'Correlation coefficient: {corr_coeff_south_summer:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1900, 4, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    plt.grid()

    plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/mean_hemisphere_plots")
    
    return None

plot_Hemisphere_Temperature_Avg(year_start, mean_north_hemisphere_winter, mean_north_hemisphere_summer, mean_south_hemisphere_winter, mean_south_hemisphere_summer)
#%%
#================MACHINE LEARNING RESSOURCES================#

#%%
from tqdm import tqdm
import ML

# Le problème s'écrit ainsi :
    # Y = optimizer.T @ X avec : (Y = time_matrix)
        # - Y la matrice des années correspondantes à la représentation des températures X
        # - X la matrice de la carte des températures mondiale
        # - optimizer : le résultat du gradient, qui fait le lien entre X et Y
    # De fait on cherche : X(t = t_2023 + dt) = optimizer.T-1 @ Y

def ML_ressources():
    time_matrix = np.ones([len(longitude), 1]) * l_time[0]
    for k in tqdm(range(1, t)):
        matrix_time_plus_1 = np.ones([len(longitude), 1]) * l_time[k]
        time_matrix = np.concatenate([time_matrix, matrix_time_plus_1], axis=0)
    
    time_matrix_square = time_matrix.reshape((2076, 360))
    
    mean = []
    for k in tqdm(range(t)):
        mean.append(np.nanmean(np.array(dict_temp[k])))
        
    return mean, time_matrix, time_matrix_square

mean, time_matrix, time_matrix_square = ML_ressources()

#%%
#================MACHINE LEARNING================#

#%%
def compute_Global_Avg_Temperature_Gradient():
    X = np.array(l_time).reshape(t, 1)
    X, m, s = ML.featureNormalize(X)
    Y = np.array(mean)
    X = np.concatenate([np.ones((t, 1)), X], axis=1)
    
    theta = np.zeros(2)
    alpha = 0.01
    num_iters = 100
    theta, J_history = ML.gradientDescent(X, Y, theta, alpha, num_iters)
    
    theta_exact = ML.normalEqn(X, Y)
    return X, Y, theta, theta_exact

#X is the colomn vector of dates, Y is the world mean temperature year by year associated
#They both are normalized, and X features a first row of ones
X, Y, theta, theta_exact = compute_Global_Avg_Temperature_Gradient()

#%%
def plot_Global_Avg_Temperature_Gradient(X, Y, theta, theta_exact, dt=100):
    fig = plt.figure(figsize=(7, 9))
    fig.subplots_adjust(hspace=0.6)
    
    #SGD regression of the temperature mean between 1850 and 2020
    plt.subplot(4, 1, 1)
    plt.plot(l_time, mean)
    plt.plot(l_time, X@theta)
    plt.title("SGD regression line between the year and the average temperature")
    corr_coeff_theta, a = pearsonr(mean, X@theta)
    textstr = f'Correlation coefficient: {corr_coeff_theta:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1845, 1.5, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    
    #Exact regression of the temperature mean between 1850 and 2020
    plt.subplot(4, 1, 2)
    plt.plot(l_time, mean)
    plt.plot(l_time, X@theta_exact)
    plt.title("Exact gradient regression line between the year and the average temperature")
    corr_coeff_theta_exact, b = pearsonr(mean, X@theta_exact)
    textstr = f'Correlation coefficient: {corr_coeff_theta_exact:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1845, 1.5, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    
    print(corr_coeff_theta_exact-corr_coeff_theta)
    print(a, b)
    
    #Prediction of the future using the SGD regression line
    dt = 100
    m = theta.shape[0]
    plt.subplot(4, 1, 3)
    plt.plot(l_time, mean)
    x_vals_ax = np.linspace(l_time[0], l_time[-1]+dt, t+dt).reshape((t+dt, 1))
    x_vals_1, m, s = ML.featureNormalize(x_vals_ax)
    x_vals_1 = np.concatenate([np.ones((t+dt, 1)), x_vals_1], axis=1)
    y_vals_1 = x_vals_1@theta
    plt.plot(x_vals_ax, y_vals_1)
    
    dt = 100
    plt.subplot(4, 1, 4)
    plt.plot(l_time, mean)
    x_vals_ax = np.linspace(l_time[0], l_time[-1]+dt, t+12*dt).reshape((t+12*dt, 1))
    print(x_vals_ax[0:10].T, l_time[0:10])
    x_vals_2, m, s = ML.featureNormalize(x_vals_ax)
    x_vals_2 = np.concatenate([np.ones((t+12*dt, 1)), x_vals_2], axis=1)
    y_vals_2 = x_vals_2@theta_exact
    plt.plot(x_vals_ax, y_vals_2)
    
    plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/global_average_temperature_gradient_plot")

    return None

plot_Global_Avg_Temperature_Gradient(X, Y, theta, theta_exact, dt=100)

#%%
def plot_Global_Avg_Temperature_Interpolation():
    fig = plt.figure(figsize=(7, 9))
    fig.subplots_adjust(hspace=0.7)
    
    #Interpolation of the temperature mean between 1850 and 2020 (order=3)
    plt.subplot(4, 1, 1)
    plt.plot(l_time, mean)
    coeffs = np.polyfit(l_time, mean, 3)
    x_vals = np.linspace(l_time[0], l_time[-1], t)
    y_vals = np.polyval(coeffs, x_vals)
    plt.plot(x_vals, y_vals)
    plt.title("Interpolation of the temperature mean between 1850 and 2020\n(order=3)")
    corr_coeff_order_3 = pearsonr(mean, y_vals)[0]
    textstr = f'Correlation coefficient: {corr_coeff_order_3:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1845, 1.5, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    
    #Interpolation of the temperature mean between 1850 and 2020 (order=5)
    plt.subplot(4, 1, 2)
    plt.plot(l_time, mean)
    coeffs_2 = np.polyfit(l_time, mean, 5)
    x_vals_2 = np.linspace(l_time[0], l_time[-1], t)
    y_vals_2 = np.polyval(coeffs_2, x_vals_2)
    plt.plot(x_vals, y_vals)    
    plt.title("Interpolation of the temperature mean between 1850 and 2020\n(order=5)")
    corr_coeff_order_5 = pearsonr(mean, y_vals_2)[0]
    textstr = f'Correlation coefficient: {corr_coeff_order_5:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1845, 1.5, textstr, fontsize=12, verticalalignment='top', 
             bbox=dict(boxstyle='square', facecolor='white'))
    
    
    #Extrapolating data to the past and the future (should I use deep learning later on ?)
    plt.subplot(4, 1, 3)
    dt = 100
    x_vals_3 = np.linspace(l_time[0], l_time[-1]+dt, t+dt)
    y_vals_3 = np.polyval(coeffs, x_vals_3)
    
    plt.plot(l_time, mean)
    plt.plot(x_vals_3, y_vals_3, label='Prediction curve')
    highlight_value = 5
    index_highlight_value = np.abs(y_vals_3 - highlight_value).argmin()
    plt.scatter(x_vals_3[index_highlight_value], highlight_value, color='black', marker='x', zorder=3)
    plt.annotate(f"+5°C reached \n in year {math.trunc(x_vals_3[index_highlight_value]*100)/100}", 
                 (index_highlight_value, 5), xytext=(2050, 4.2), ha='center')
    plt.title("Extrapolation of the temperature mean between 1850 and 2120\n(order=3)")
    plt.grid()
    
    #Extrapolation of the temperature mean between 1850 and 2120
    plt.subplot(4, 1, 4)
    dt = 50
    x_vals_4 = np.linspace(l_time[0], l_time[-1]+dt, t+dt)
    y_vals_4 = np.polyval(coeffs_2, x_vals_4)
    
    plt.plot(l_time, mean)
    plt.plot(l_time, mean, color='C0')
    plt.plot(x_vals_4, y_vals_4, label='Prediction curve', color='C1')
    highlight_value_2 = 5
    index_highlight_value_2 = np.abs(y_vals_4 - highlight_value_2).argmin()
    plt.scatter(x_vals_4[index_highlight_value_2], highlight_value_2, color='black', marker='x', zorder=3)
    plt.annotate(f"+5°C reached \n in year {math.trunc(x_vals_4[index_highlight_value_2]*100)/100}", 
                 (index_highlight_value_2, 5), xytext=(2025, 4.2), ha='center')
    plt.title("Extrapolation of the temperature mean between 1850 and 2080\n(order=5)")
    plt.grid()
    
    plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/global_average_temperature_interpolation_plot")

    return None

plot_Global_Avg_Temperature_Interpolation()

#%%
#================TENSORFLOW================#

#%%

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras

dict_temp_matrix = np.array(dict_temp)

#%%
#Useful functions to improve calculation speed and solve NaN issues when computing the mean
def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std

def nan_to_0(X):
    (l, m, n) = np.shape(X)
    for k in tqdm(range(l)):
        for i in range(m):
            for j in range(n):
                if np.isnan(X[k, i, j]):
                    X[k, i, j] = 0
    return X

#%%
#X is the temperature variation matrix where NaN data is changd to 0
#Y is the corresponding year and month vector 
#It should be tested with Y a time vector where 12 months are regrouped as 1 year
#Y_year = math.trunc(Y)
X = normalize(nan_to_0(dict_temp_matrix))
Y = np.array([l_time]).T

#%%
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras

print(X.shape, Y.shape)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(180, 360)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(t)
])
print(model.summary())

#loss and optimizer definition                   
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.01)
metrics = ['accuracy']
model.compile(loss=loss, optimizer=optim, metrics=metrics)

#Training calculation time :
    #bz=10 : ~500sec
    #bz=t : ~20sec
    #e=5 ~ e=20 : no impact

batch_size = t
epochs = 20

model.fit(X, Y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

#evaluation
model.evaluate(X[0:100], Y[0:100], batch_size=batch_size, verbose=2)

#%%
model.save('C:/Travail/IPSA/Aero4/Module électif/Neuronal_network_Temperature_Year_128l_20e_2076bz')
#%%
model = tf.keras.models.load_model('C:/Travail/IPSA/Aero4/Module électif/Neuronal_network_Temperature_Year_128l_20e_2076bz')
#%%
#Prediction and testing
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
    ])

X_test = X
predictions_1 = (probability_model(X_test)).numpy()

predict_year_1 = []
n = np.shape(predictions_1)[0]
for k in range(n):
    predict_year_1.append(np.argmax(predictions_1[k]))

plt.plot(predict_year_1, '.')
if n <= 241:
    xticks = np.linspace(0, 12*(n//12), n//12+1)
    plt.xticks(xticks)

#%%
plt.savefig("C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/Neurone_128l_20e_10bz")

#%%

predictions_2 = model(X_test)
predictions_2 = tf.nn.softmax(predictions_2)
predictions_2 = predictions_2.numpy()

predict_year_2 = []
for k in range(n):
    predict_year_2.append(np.argmax(predictions_2[k]))

plt.plot(predict_year_2, '.')