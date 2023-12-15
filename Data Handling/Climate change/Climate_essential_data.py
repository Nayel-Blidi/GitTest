"""
This dataset contains the following products calculated from the ERA5 and ERA-Interim data:

    monthly means
    monthly anomalies with respect to 1981-2010 climatologies
    12 months running mean anomalies with respect to 1981-2010 climatologies
    
for the following ECVs:

    Surface air temperature
    Total precipitation
    Volumetric soil moisture for the uppermost layer of soil 0-7 cm (the percentage of water per unit volume)
    Surface air relative humidity
    Sea-ice cover
"""
#https://cds.climate.copernicus.eu/cdsapp#!/dataset/ecv-for-climate-change?tab=overview
#https://confluence.ecmwf.int/display/CKB/Essential+Climate+Variables+for+assessment+of+climate+variability+from+1979+to+present%3A+Product+user+guide#EssentialClimateVariablesforassessmentofclimatevariabilityfrom1979topresent:Productuserguide-Acronyms

import xarray as xr
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ML

#%% 
#================LOADING DATASETS================#

#%%
#Anomaly of the month
data_0_t2m = xr.open_dataset('D:/Datasets_ecv/dataset_1month_0.nc')['t2m']
data_2_r = xr.open_dataset('D:/Datasets_ecv/dataset_1month_2.nc')['r']
data_3_swvl1 = xr.open_dataset('D:/Datasets_ecv/dataset_1month_3.nc')['swvl1']
data_4_tp = xr.open_dataset('D:/Datasets_ecv/dataset_1month_4.nc')['tp']
data_4_tp = data_4_tp.reindex_like(data_0_t2m, method='nearest')

#Mean of the month
data_5_t2m = xr.open_dataset('D:/Datasets_ecv/dataset_1month_5.nc')['t2m']
data_7_r = xr.open_dataset('D:/Datasets_ecv/dataset_1month_7.nc')['r']
data_8_swvl1 = xr.open_dataset('D:/Datasets_ecv/dataset_1month_8.nc')['swvl1']
data_9_tp = xr.open_dataset('D:/Datasets_ecv/dataset_1month_9.nc')['tp']
data_9_tp = data_9_tp.reindex_like(data_5_t2m, method='nearest')

#Time periods
time_t2m = [s[0:10] for s in np.datetime_as_string(data_5_t2m['time'].values)]
time_r = [s[0:10] for s in np.datetime_as_string(data_7_r['time'].values)]
time_swvl1 = [s[0:10] for s in np.datetime_as_string(data_8_swvl1['time'].values)]
time_tp =[s[0:10] for s in np.datetime_as_string(data_9_tp['time'].values)]
time = time_t2m.copy()
t = len(time)

#%% 
#================PLOT DATASETS================#

#%% load_average_eau
from tqdm import tqdm
import xarray as xr

def load_average_eau(data_5_t2m, data_7_r, data_8_swvl1, data_9_tp):
    
    time_mean = []
    time_1979 = []
    time_2022 = []
    
    mean_mean_t2m = data_5_t2m.mean(dim='time')
    mean_mean_t2m = mean_mean_t2m.values
    time_mean.append('Temperature 2m high')
    time_1979.append(np.datetime_as_string(data_5_t2m['time'].values[11])[0:-10])
    time_2022.append(np.datetime_as_string(data_5_t2m['time'].values[-1])[0:-10])
    
    mean_mean_r = data_7_r.mean(dim='time')
    mean_mean_r = mean_mean_r.values
    time_mean.append('Relative air humidity')
    time_1979.append(np.datetime_as_string(data_7_r['time'].values[11])[0:-10])
    time_2022.append(np.datetime_as_string(data_7_r['time'].values[-1])[0:-10])
    
    mean_mean_swvl1 = data_8_swvl1.mean(dim='time')
    mean_mean_swvl1 = mean_mean_swvl1.values
    time_mean.append('Volumetric soil water cm deep')
    time_1979.append(np.datetime_as_string(data_8_swvl1['time'].values[11])[0:-10])
    time_2022.append(np.datetime_as_string(data_8_swvl1['time'].values[-1])[0:-10])
    
    mean_mean_tp = data_9_tp.mean(dim='time')
    mean_mean_tp = mean_mean_tp.values
    time_mean.append('Total precipitations')
    time_1979.append(np.datetime_as_string(data_9_tp['time'].values[11])[0:-10])
    time_2022.append(np.datetime_as_string(data_9_tp['time'].values[-1])[0:-10])
    
    datasets_list_1979 = [data_5_t2m[0, :, :], data_7_r[0, :, :], 
                          data_8_swvl1[0, :, :], data_9_tp[0, :, :]]
    datasets_list_2019 = [data_5_t2m[-1, :, :], data_7_r[-1, :, :],
                          data_8_swvl1[-1, :, :], data_9_tp[-1, :, :]]    
    mean_list = [mean_mean_t2m, mean_mean_r, mean_mean_swvl1, mean_mean_tp]
    
    datasets_plot = mean_list + datasets_list_1979 + datasets_list_2019
    time_plot = time_mean + time_1979 + time_2022 
    
    return datasets_plot, time_plot

datasets_plot, time_plot = load_average_eau(data_5_t2m, data_7_r, data_8_swvl1, data_9_tp)

#%% plot_average_eau
def plot_average_eau(datasets_plot, time_plot):
    
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(16, 7))
    for k in range(12):
        plt.subplot(3, 4, k+1)
        im = plt.imshow(datasets_plot[k])
        plt.title(time_plot[k])
    
    
    plt.suptitle("Highlight of average humidity related ECV", fontsize=20)
    plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/ECV_1979_2022.png')
    return 0

plot_average_eau(datasets_plot, time_plot)

#%% t2m_variations

def t2m_variations(data_0_t2m):
    
    time=[]
    mean_mean_t2m = data_0_t2m.mean(dim='time')
    mean_mean_t2m = mean_mean_t2m.values
    time.append('Average temperature variation 2m high')
    time.append(np.datetime_as_string(data_0_t2m['time'].values[11])[0:-10])
    time.append(np.datetime_as_string(data_0_t2m['time'].values[-1])[0:-10])
    
    data_plot = [mean_mean_t2m, data_0_t2m[0, :, :], data_0_t2m[-1, :, :]]
    
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
    for k in range(3):
        plt.subplot(3, 1, k+1)
        im = plt.imshow(data_plot[k])
        plt.title(time[k])
    
    colorbar = fig.colorbar(im, ax=ax, orientation='vertical')
    colorbar.ax.text(2.5, 0.4, 'Variations (°K)', transform=colorbar.ax.transAxes, rotation=90)

    plt.suptitle("Temperature variations 1979-2018", fontsize=14)
    plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/ECV_t2m_variations.png')

    return 0
    
t2m_variations(data_0_t2m)

#%% r_variations

def r_variations(data_2_r):
    
    time=[]
    mean_mean_r = data_2_r.mean(dim='time')
    mean_mean_r = mean_mean_r.values
    time.append('Average relative air humidity')
    time.append(np.datetime_as_string(data_2_r['time'].values[11])[0:-10])
    time.append(np.datetime_as_string(data_2_r['time'].values[-1])[0:-10])
    
    data_plot = [mean_mean_r, data_2_r[0, :, :], data_2_r[-1, :, :]]
    
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
    for k in range(3):
        plt.subplot(3, 1, k+1)
        im = plt.imshow(data_plot[k])
        plt.title(time[k])

    colorbar = fig.colorbar(im, ax=ax, orientation='vertical')
    colorbar.ax.text(2.5, 0.4, 'Variations (%)', transform=colorbar.ax.transAxes, rotation=90)

    plt.suptitle("Relative air humidity 1979-2018", fontsize=14)
    plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/ECV_r_variations.png')

    return 0
    
r_variations(data_2_r)

#%% swvl1_variations

def swvl1_variations(data_3_swvl1):
    
    time=[]
    mean_mean_swvl1 = data_3_swvl1.mean(dim='time')
    mean_mean_swvl1 = mean_mean_swvl1.values
    time.append('Average soil humidity variations 7cm deep')
    time.append(np.datetime_as_string(data_3_swvl1['time'].values[11])[0:-10])
    time.append(np.datetime_as_string(data_3_swvl1['time'].values[-1])[0:-10])
    
    data_plot = [mean_mean_swvl1, data_3_swvl1[0, :, :], data_3_swvl1[-1, :, :]]
    
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
    for k in range(3):
        plt.subplot(3, 1, k+1)
        im = plt.imshow(data_plot[k])
        plt.title(time[k])

    colorbar = fig.colorbar(im, ax=ax, orientation='vertical')
    colorbar.ax.text(2.5, 0.4, 'Variations (m3.m-3)', transform=colorbar.ax.transAxes, rotation=90)

    plt.suptitle("Soil humidity 1979-2018", fontsize=14)
    plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/ECV_soil_variations.png')

    return 0
    
swvl1_variations(data_3_swvl1)

#%% tp_variations

def tp_variations(data_4_tp):
    
    time=[]
    mean_mean_tp = data_4_tp.mean(dim='time')
    mean_mean_tp = mean_mean_tp.values
    time.append('Average total precipitation')
    time.append(np.datetime_as_string(data_4_tp['time'].values[11])[0:-10])
    time.append(np.datetime_as_string(data_4_tp['time'].values[-1])[0:-10])
    
    data_plot = [mean_mean_tp, data_4_tp[0, :, :], data_4_tp[-1, :, :]]
    
    fig, ax = plt.subplots(3, 1, figsize=(6, 9))
    for k in range(3):
        plt.subplot(3, 1, k+1)
        im = plt.imshow(data_plot[k])
        plt.title(time[k])

    colorbar = fig.colorbar(im, ax=ax, orientation='vertical')
    colorbar.ax.text(2.5, 0.4, 'Variations (m.day-1)', transform=colorbar.ax.transAxes, rotation=90)

    plt.suptitle("Precipitation variations 1979-2018", fontsize=14)
    plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS/ECV_precipitation_variations.png')

    return 0
    
tp_variations(data_4_tp)

#%% 
#================EDIT DATASETS================#

#%%
#Est notable :
    #Temperature : pôle nord plus chaud hors islandis, antartic semble se refroidir
    #Relative humidity : assèchements de l'australie, du pérou, de l'afrique du sud, 
    #                   du moyen orient, de steppes mongoles, de la côte ouest des US
    #Humidité des sols : assèchement de l'afrique du sud, de l'autralie, de la côte ouest des US
    #                   humidification du nord de l'inde, du nord de l'amérique du sud
    #Précipitations : Moyenne constante, pas de variation, dur à lire, cf plus localement
#Confirmé par les plots des variations, quoique moins lisisble
    #Pour les températures, quoique réchauffement du coeur de l'antartic
    #pour l'humidité relative
    #humidité des sols peu lisible
    #précipitations non lisibles
    
#Etude locale : moyen orient, côte ouest US, côtes ouest Pérou, afrique du sud ?

#créer un masque terrestre pour calculer des moyennes : comparaison du morient et us ?

#%% Coordonées pays étudiés
longitude_morient = [100, 300]
latitude_morient = [180, 300]

longitude_us = [900, 1100]
latitude_us = [150, 350]

longitude_perou= [1120,1200]
latitude_perou = [400, 600]

longitude_sa = [0, 220]
latitude_sa = [300, 530]

#%% 
#================MACHINE LEARNING================#

#%%
#Coordonées pays étudiés : Perou
lon_mo_min, lon_mo_max = 1120, 1200
lat_mo_min, lat_mo_max = 400, 600

longitude_sa = [0, 220]
latitude_sa = [300, 530]

# 4D array of ECV monthly means
array_anomaly = xr.concat([data_0_t2m[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max], 
                            data_2_r[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                            dim='data_vect')

array_anomaly = xr.concat([array_anomaly,
                            data_3_swvl1[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                            dim='data_vect')

# Get rid of additional coordinate data, that obstruct the datsets concatenation
array_anomaly_bis = array_anomaly.drop('step')
array_anomaly_bis = array_anomaly_bis.drop('valid_time')
array_anomaly_bis = array_anomaly_bis.drop('surface')
array_anomaly_bis = array_anomaly_bis.drop('depthBelowLandLayer')

data_4_tp_bis = data_4_tp.drop('step')
data_4_tp_bis = data_4_tp_bis.drop('valid_time')
data_4_tp_bis = data_4_tp_bis.drop('surface')

array_anomaly = xr.concat([array_anomaly_bis, 
                            data_4_tp[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                            dim='data_vect')
metadata_anomaly = array_anomaly.attrs
del array_anomaly_bis

# 4D array of ECV monthly anomalies
array_mean = xr.concat([data_5_t2m[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max], 
                        data_7_r[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                        dim='data_vect')

array_mean = xr.concat([array_mean,
                        data_8_swvl1[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                        dim='data_vect')

# Get rid of additional coordinate data, that obstruct the datsets concatenation
array_mean_bis = array_mean.drop('step')
array_mean_bis = array_mean_bis.drop('valid_time')
array_mean_bis = array_mean_bis.drop('surface')

data_9_tp_bis = data_9_tp.drop('step')
data_9_tp_bis = data_9_tp_bis.drop('valid_time')
data_9_tp_bis = data_9_tp_bis.drop('surface')

array_mean = xr.concat([array_mean_bis, 
                        data_9_tp[:, lat_mo_min:lat_mo_max, lon_mo_min:lon_mo_max]],
                        dim='data_vect')
metadata_mean = array_mean.attrs
del array_mean_bis

#Masking array without oceans

array_anomaly_masked = array_anomaly.copy()
array_mean_masked = array_mean.copy()
array_anomaly_0 = array_anomaly.copy()
array_mean_0 = array_mean.copy()
for i in tqdm(range(lon_mo_max-lon_mo_min)):
    for j in range(lat_mo_max-lat_mo_min):
        if np.isnan(array_anomaly[2,0,j,i]):
            #Plot arrays, land only
            array_anomaly_masked[:,:,j,i] = np.nan
            array_mean_masked[:,:,j,i] = np.nan
            #ML corrected soil humidity arrays
            array_anomaly_0[2,:,j,i] = 0
            array_mean_0[2,:,j,i] = 0
            
#Computation of ground evolution
t2m_mean = []
r_mean = []
swvl1_mean = []
tp_mean = []

t2m_anomaly = []
r_anomaly = []
swvl1_anomaly = []
tp_anomaly = []

for t in tqdm(range(len(time))):
    t2m_mean.append(np.nanmean(array_mean_masked[0, t, :, :]))
    r_mean.append(np.nanmean(array_mean_masked[1, t, :, :]))
    swvl1_mean.append(np.nanmean(array_mean_masked[2, t, :, :]))
    tp_mean.append(np.nanmean(array_mean_masked[3, t, :, :]))

    t2m_anomaly.append(np.nanmean(array_anomaly_masked[0, t, :, :]))
    r_anomaly.append(np.nanmean(array_anomaly_masked[1, t, :, :]))
    swvl1_anomaly.append(np.nanmean(array_anomaly_masked[2, t, :, :]))
    tp_anomaly.append(np.nanmean(array_anomaly_masked[3, t, :, :]))

#%%
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras

tt=120
X = np.ones((tt, 230, 220))
for k in range(tt):
    X[k, :, :] *= k

X = np.linspace(0, tt, tt)

Y = array_mean_0.values[0,:tt,:,:].reshape((tt, 230, 220))
for k in range(0, tt, 12):
    plt.subplot(4, 3, int(k/12+1))
    plt.imshow(Y[k, :, :])
    

#X = data_0_t2m[: ,:, :].copy()
print(X.shape, Y.shape)
#X.reshape((480, 4, 230, 220))

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((tt,1), input_shape=(tt,)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.Reshape((98, 128, 1)),
    tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), activation='relu', padding='valid'),
    tf.keras.layers.Reshape((96, 126, 1)),
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
epochs = 10

model.fit(X, Y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

#evaluation
model.evaluate(X[0:100], Y[0:100], batch_size=batch_size, verbose=2)

#%% 
#================MACHINE LEARNING 2================#

#%%
import ML

t = len(time)
time_idx = np.linspace(1979, 2019, t)

means = np.concatenate([np.array([t2m_mean]), np.array([r_mean]), np.array([swvl1_mean]), np.array([tp_mean])], axis = 0)
means = means.T

#%%
plt.plot(time_idx, means[:,0])
plt.plot(time_idx, means[:,1])
plt.plot(time_idx, means[:,2])
plt.plot(time_idx, means[:,3])
plt.semilogy(time_idx, means[:,0])

mois = np.array([time_idx]).T
mois = ML.featureNormalize(mois)[0]
mois = np.concatenate([np.ones((t, 1)), mois], axis=1)
temperature = means[:,0]
theta = np.random.rand(mois.shape[1])

theta_, J = ML.gaussianGradient(mois, temperature, theta, alpha=0.1, num_iters=10, sigma=1)

#%%
plt.plot(time_idx, means[:,0])
test_temperature = mois@theta_
plt.plot(time_idx, mois@theta_)

plt.figure()
time_idx_bis = np.linspace(1979, 2049, t*2)
mois_bis = np.concatenate([np.ones((t*2, 1)), ML.featureNormalize(np.array([time_idx_bis]).T)[0]], axis=1)
plt.plot(time_idx, means[:,0])
plt.plot(time_idx_bis, mois_bis@theta_)
#%%
x_ecv = ML.featureNormalize(means)[0].reshape(t, 4)
y_annees = np.array([time_idx]).T
theta = np.random.rand(x_ecv.shape[1])

plt.plot(time_idx, x_ecv[:,0])
plt.plot(time_idx, x_ecv[:,1])
plt.plot(time_idx, x_ecv[:,2])
plt.plot(time_idx, x_ecv[:,3])

#%%
theta_, J = ML.gaussianGradient(x_ecv, y_annees, theta, alpha=1, num_iters=10, sigma=1)

test_annees = x_ecv@theta_
plt.plot(time_idx, test_annees)
plt.plot(time_idx, x_ecv[:,0])
plt.plot(time_idx, x_ecv[:,1])
plt.plot(time_idx, x_ecv[:,2])
plt.plot(time_idx, x_ecv[:,3])

#%%
import ML

sample_size = 480

x_ecv = ML.featureNormalize(means)[0].reshape(t, 4)
y_annees = np.array([time_idx]).T
y_mois = []
for k in range(sample_size):
    y_mois.append(k%12+1)
y_mois=np.array(y_mois)

theta = np.random.rand(x_ecv.shape[1])

#%%

Theta1, Theta2 = ML.nnOneLayer(x_ecv, y_mois, num_labels=12)

prediction = ML.predOneLayer(x_ecv, Theta1, Theta2)

print('nnOneLayer Training Accuracy : %.2f' % (np.mean(prediction == y_mois) * 100), "%")

#%%
import numpy as np
from scipy.optimize import minimize

# define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidInv(sigmoid):
    return -np.log(1/sigmoid + 1)

# define the feedforward neural network function
def neural_network(Theta1, Theta2, X):
    W1, W2 = Theta1, Theta2
    z1 = X.dot(W1)
    a1 = sigmoid(z1)
    z2 = a1.dot(W2)
    y = sigmoid(z2)
    return y

# define the objective function to minimize
def objective_function(X, theta, y_target):
    y_pred = neural_network(theta, X)
    error = np.sum((y_pred - y_target)**2)
    return error

# set the target output of the neural network
y_target = prediction

# set the weights and biases of the neural network
W1 = Theta1
b1 = np.zeros(W1.shape)
W2 = Theta2
b2 = np.zeros(Theta2.shape)
thetas = [W1, W2]

# set an initial guess for the input
X_guess = np.ones(x_ecv.shape)/2

# use the minimize function to find the input that minimizes the objective function
res = minimize(objective_function, X_guess, args=(thetas, y_target), method='BFGS')

# print the solution
print("Input that produces the target output:", res.x)


#%%

def sigmoidInv(sigmoid):
    return -np.log(1/sigmoid + 1)

"""
W1, W2 = Theta1, Theta2
z1 = x_ecv.dot(W1)
a1 = sigmoid(z1)
z2 = a1.dot(W2)
y = sigmoid(z2)
"""

m = x_ecv.shape[0]

a1 = np.concatenate([np.ones((m, 1)), x_ecv], axis=1)
z2 = a1 @ Theta1.T
a2 = np.concatenate([np.ones((m, 1)), 1 / (1 + np.exp(-z2))], axis=1)
h = 1 / (1 + np.exp(-a2 @ Theta2.T))

z2_inv = sigmoidInv(h)
a1_inv = z2_inv @ np.linalg.inv(Theta2)
z1_inv = sigmoidInv(a1_inv)
X_inv = z1_inv @ np.linlagl.inv(Theta1)











