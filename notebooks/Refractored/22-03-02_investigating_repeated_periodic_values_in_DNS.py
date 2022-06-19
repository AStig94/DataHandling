#%%
import xarray as xr
from DataHandling import utility
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import os

slice_array=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")

#%% 
Re_Tau = 395
Re = 10400          #Direct from simulation
nu = 1/Re           #Kinematic viscosity
u_tau = Re_Tau*nu   
Q_avg=0.665

u_vel=slice_array['u_vel'].isel(time=-1).sel(y=utility.y_plus_to_y(15),method="nearest")

#%% Test if x=0 and x=12 are equal
x_start = u_vel.isel(x=0).values
x_end = u_vel.isel(x=-1).values
x_comparison = x_start==x_end
x_equal_arrays = x_comparison.all()
print('Are there repeated periodic values in x? Answer:',x_equal_arrays)

#%% Test if z=-3 and z=3 are equal
z_start = u_vel.isel(x=0).values
z_end = u_vel.isel(x=-1).values
z_comparison = z_start==z_end
z_equal_arrays = z_comparison.all()
print('Are there repeated periodic values in z? Answer:',z_equal_arrays)

#%%
