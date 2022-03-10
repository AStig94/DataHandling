#%%

from DataHandling import utility
import xarray as xr



df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
slice_array=df


var=['u_vel']
target=['tau_wall']
y_plus=15
normalized=False
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity

#%%
var.append(target[0])
if target[0]=='tau_wall':
    target_slice1=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
    target_slice1=nu*target_slice1
    
    #target_slice2=slice_array['u_vel'].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
    #target_slice2=nu*target_slice2
    
    if normalized==True:
        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))

#Checking if the target contains _flux
elif target[0][-5:] =='_flux':
    target_slice1=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
    pr_number=float(target[0][2:-5])
    target_slice1=nu/(pr_number)*target_slice1
    
    #target_slice2=slice_array[target[0][:-5]].differentiate('y').sel(y=slice_array['y'].max(),method="nearest")
    #target_slice2=nu/(pr_number)*target_slice2
    
    if normalized==True:
        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))
else:
    target_slice1=slice_array[target[0]].sel(y=utility.y_plus_to_y(0),method="nearest")

    #target_slice2=slice_array[target[0]].sel(y=slice_array['y'].max(),method="nearest")
    if normalized==True:
        target_slice1=(target_slice1-target_slice1.mean(dim=('time','x','z')))/(target_slice1.std(dim=('time','x','z')))

#%%

wall_1=slice_array.sel(y=utility.y_plus_to_y(y_plus),method="nearest")
wall_1[target[0]]=target_slice1
wall_1=wall_1[var]  # Remember target is appended to var further up
# %%
y_plus2 = 30
var = ['u_vel']
wall_2=slice_array.sel(y=utility.y_plus_to_y(y_plus2),method="nearest")

wall_2=wall_2[var]
var2=['u_vel_30']
wall_2=wall_2.rename({var[0]:var2[0]})
# %%
