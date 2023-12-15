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

#Anomaly of the month
data_0_t2m = xr.open_dataset('D:/Datasets_fit/dataset_1month_0.nc')['t2m']
data_2_r = xr.open_dataset('D:/Datasets_fit/dataset_1month_2.nc')['r']
data_3_swvl1 = xr.open_dataset('D:/Datasets_fit/dataset_1month_3.nc')['swvl1']
data_4_tp = xr.open_dataset('D:/Datasets_fit/dataset_1month_4.nc')['tp']
data_4_tp = data_4_tp.reindex_like(data_0_t2m, method='nearest')

#Mean of the month
data_5_t2m = xr.open_dataset('D:/Datasets_fit/dataset_1month_5.nc')['t2m']
data_7_r = xr.open_dataset('D:/Datasets_fit/dataset_1month_7.nc')['r']
data_8_swvl1 = xr.open_dataset('D:/Datasets_fit/dataset_1month_8.nc')['swvl1']
data_9_tp = xr.open_dataset('D:/Datasets_fit/dataset_1month_9.nc')['tp']
data_9_tp = data_9_tp.reindex_like(data_5_t2m, method='nearest')

#Time periods
time_t2m = [s[0:-10] for s in np.datetime_as_string(data_5_t2m['time'].values)]
time_r = [s[0:-10] for s in np.datetime_as_string(data_7_r['time'].values)]
time_swvl1 = [s[0:-10] for s in np.datetime_as_string(data_8_swvl1['time'].values)]
time_tp =[s[0:-10] for s in np.datetime_as_string(data_9_tp['time'].values)]
time = time_t2m.copy()

#%%
#Coordonées pays étudiés
lon_mo_min, lon_mo_max = 1120, 1200
lat_mo_min, lat_mo_max = 400, 600

longitude_perou= [1120,1200]
latitude_perou = [400, 600]

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
for i in tqdm(range(lon_mo_max-lon_mo_min)):
    for j in range(lat_mo_max-lat_mo_min):
        if np.isnan(array_anomaly[2,0,j,i]):
            array_anomaly_masked[:,:,j,i] = np.nan
            array_mean_masked[:,:,j,i] = np.nan

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
plt.subplot(2,2,1)
plt.imshow(array_anomaly.values[0, 0, :,:])
plt.title("t2m \n"+time[0])
plt.subplot(2,2,2)
plt.imshow(array_anomaly.values[1, 0, :,:])
plt.title("r \n"+time[0])
plt.subplot(2,2,3)
plt.imshow(array_anomaly.values[2, 0, :,:])
plt.title("swvl1 \n"+time[0])
plt.subplot(2,2,4)
plt.imshow(array_anomaly.values[3, 0, :,:])
plt.title("tp \n"+time[0])
plt.suptitle('Perou_1979')
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_1979')

plt.figure()
plt.subplot(2,2,1)
plt.imshow(array_anomaly.values[0, -1, :,:])
plt.title("t2m \n"+time[-1])
plt.subplot(2,2,2)
plt.imshow(array_anomaly.values[1, -1, :,:])
plt.title("r \n"+time[-1])
plt.subplot(2,2,3)
plt.imshow(array_anomaly.values[2, -1, :,:])
plt.title("swvl1 \n"+time[-1])
plt.subplot(2,2,4)
plt.imshow(array_anomaly.values[3, -1, :,:])
plt.title("tp \n"+time[-1])
plt.suptitle('Perou_2019')
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_2019')

#%%
plt.close('all')

fig, ax = plt.subplots(4, 10)
plt.suptitle("Yearly t2m mean anomaly \n [-3°K, +3°K]")
for k in range(40):
    plt.subplot(4, 10,k+1)
    plt.imshow(array_anomaly[0, k*12:(k+1)*12, :, :].mean('time'), vmin=-3, vmax=3)
    plt.title(time[k*12], fontsize=8)
    plt.axis('off')
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_t2m_anomaly_evo')

fig, ax = plt.subplots(4, 10)
plt.suptitle("Yearly r mean anomaly \n [-10%, +10%]")
for k in range(40):
    plt.subplot(4, 10,k+1)
    plt.imshow(array_anomaly[1, k*12:(k+1)*12, :, :].mean('time'), vmin=-10, vmax=10)
    plt.title(time[k*12], fontsize=8)
    plt.axis('off')
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_r_anomaly_evo')

fig, ax = plt.subplots(4, 10)
plt.suptitle("Yearly swvl1 mean anomaly \n [-0.1 m3.m-3, +0.1 m3.m-3]")
for k in range(40):
    plt.subplot(4, 10,k+1)
    plt.imshow(array_anomaly[2, k*12:(k+1)*12, :, :].mean('time'), vmin=-0.1, vmax=0.1)
    plt.title(time[k*12], fontsize=8)
    plt.axis('off')
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_swvl1_anomaly_evo')

fig, ax = plt.subplots(4, 10)
plt.suptitle("Yearly tp mean anomaly \n [-10e-4 m, 10e-4 m]")
for k in range(40):
    plt.subplot(4, 10, k+1)
    plt.imshow(array_anomaly[3, k*12:(k+1)*12, :, :].mean('time'), vmin=-10e-4, vmax=10e-4)
    plt.title(time[k*12], fontsize=8)
    plt.axis('off')
plt.get_current_fig_manager().full_screen_toggle()
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_tp_anomaly_evo')

plt.close('all')

#%% Plot of ground evolution
plt.close('all')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.suptitle("Land means evolutions : Perou & Chile")
plt.subplot(2, 2, 1)
plt.plot(time, t2m_mean)
plt.plot(time[:-12], [np.mean(t2m_mean[s:(s+12)]) for s in range(len(time)-12)])
plt.title("t2m_mean")
plt.xticks(rotation=60)
plt.subplot(2, 2, 2)
plt.plot(time, r_mean)
plt.plot(time[:-12], [np.mean(r_mean[s:(s+12)]) for s in range(len(time)-12)])
plt.title("r_mean")
plt.xticks(rotation=60)
plt.subplot(2, 2, 3)
plt.plot(time, swvl1_mean)
plt.plot(time[:-12], [np.mean(swvl1_mean[s:(s+12)]) for s in range(len(time)-12)])
plt.title("swvl1_mean")
plt.xticks(rotation=60)
plt.subplot(2, 2, 4)
plt.plot(time, tp_mean)
plt.plot(time[:-12], [np.mean(tp_mean[s:(s+12)]) for s in range(len(time)-12)])
plt.title("tp_mean")
plt.xticks(rotation=60)
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_land_means_evo')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4)
for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.suptitle("Land anomalies evolutions : Perou & Chile")
plt.subplot(2, 2, 1)
plt.plot(time, t2m_anomaly)
plt.plot(time[:-12], [np.mean(t2m_anomaly[s:(s+12)]) for s in range(len(time)-12)])
plt.title("t2m_anomaly")
plt.xticks(rotation=60)
plt.subplot(2, 2, 2)
plt.plot(time, r_anomaly)
plt.plot(time[:-12], [np.mean(r_anomaly[s:(s+12)]) for s in range(len(time)-12)])
plt.title("r_anomaly")
plt.xticks(rotation=60)
plt.subplot(2, 2, 3)
plt.plot(time, swvl1_anomaly)
plt.plot(time[:-12], [np.mean(swvl1_anomaly[s:(s+12)]) for s in range(len(time)-12)])
plt.title("swvl1_anomaly")
plt.xticks(rotation=60)
plt.subplot(2, 2, 4)
plt.plot(time, tp_anomaly)
plt.plot(time[:-12], [np.mean(tp_anomaly[s:(s+12)]) for s in range(len(time)-12)])
plt.title("tp_anomaly")
plt.xticks(rotation=60)
plt.savefig('C:/Travail/IPSA/Aero4/Module électif/GRAPHS_PNG_PLOTS_ECV/Perou_land_anomalies_evo')


















