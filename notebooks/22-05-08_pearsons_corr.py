#%%
import pandas as pd
import xarray as xr
from DataHandling import utility
import seaborn as sns
import numpy as np
import os
import matplotlib
from random import sample
#matplotlib.use('Agg')
import scipy.stats as stats

Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
save_spot=os.path.join("/home/au567859/DataHandling/data/")

ds=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")

sequence = [i for i in range(ds.sizes["time"])]
subset = sample(sequence, 100)

ds=ds.isel(time=subset)

#%% velocity

Re_Tau = 395  # Direct from simulation
Re = 10400  # Direct from simulation
nu = 1 / Re  # Kinematic viscosity
u_tau = Re_Tau * nu  # The friction velocity
Q_avg=0.665
val_list=sorted(list(ds.keys()))
val_list.remove("w_vel")
val_list.remove("u_vel")

r_list=pd.DataFrame(columns=['r_pr0.025','r_pr0.2','r_pr0.71','r_pr1'])
p_list=pd.DataFrame(columns=['p_pr0.025','p_pr0.2','p_pr0.71','p_pr1'])
PDF_data=pd.DataFrame()

y_100=ds.sel(y=utility.y_plus_to_y(100),method="nearest")["y"].values

print("data loaded, ready to process..",flush=True)
k=0
for val in val_list:
    if k>=4:
        for i in range(2,49):
            data=ds.isel(y=-i)
            y=data["y"].values
            data=data[val].values.flatten()
            #mean_val=xr.DataArray.mean(data[val],dim=('time','x','z'))
            
            #fluc = data[val]-mean_val
            #mean_fluc=xr.DataArray.mean(fluc,dim=('time','x','z')) # Should be zero

            #rms_fluc=((fluc-mean_fluc)**2)
            #rms_fluc=xr.ufuncs.sqrt(xr.DataArray.mean(rms_fluc,dim=('time','x','z')))

            #fluc_over_rms_fluc=(fluc/rms_fluc).values.flatten()
            #PDF_data[val+'_fluc_over_rmsFluc_yplus_'+str(i)]=fluc_over_rms_fluc.values.flatten()
            #print("done with y_plus=" +str(i),flush=True)

            #r, p = stats.pearsonr(fluc_over_rms_fluc, PDF_data['pr0.025_flux_fluc_over_rmsFluc'])
            r, p = stats.pearsonr(data, PDF_data['pr0.025_flux_fluc_over_rmsFluc'])
            r_list.loc[(utility.y_to_y_plus(y),"r_pr0.025")]=r
            p_list.loc[(utility.y_to_y_plus(y),"p_pr0.025")]=p

            #r, p = stats.pearsonr(fluc_over_rms_fluc, PDF_data['pr0.2_flux_fluc_over_rmsFluc'])
            r, p = stats.pearsonr(data, PDF_data['pr0.2_flux_fluc_over_rmsFluc'])
            r_list.loc[(utility.y_to_y_plus(y),"r_pr0.2")]=r
            p_list.loc[(utility.y_to_y_plus(y),"p_pr0.2")]=p

            #r, p = stats.pearsonr(fluc_over_rms_fluc, PDF_data['pr0.71_flux_fluc_over_rmsFluc'])
            r, p = stats.pearsonr(data, PDF_data['pr0.71_flux_fluc_over_rmsFluc'])
            r_list.loc[(utility.y_to_y_plus(y),"r_pr0.71")]=r
            p_list.loc[(utility.y_to_y_plus(y),"p_pr0.71")]=p

            #r, p = stats.pearsonr(fluc_over_rms_fluc, PDF_data['pr1_flux_fluc_over_rmsFluc'])
            r, p = stats.pearsonr(data, PDF_data['pr1_flux_fluc_over_rmsFluc'])
            r_list.loc[(utility.y_to_y_plus(y),"r_pr1")]=r
            p_list.loc[(utility.y_to_y_plus(y),"p_pr1")]=p
            
            print("done with y_plus="+str(utility.y_to_y_plus(y)),flush=True)

    else:
        theta_wall = (nu/(float(val[2:]))*(ds[val].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")))
        #theta_wall_fluc = theta_wall - xr.DataArray.mean(theta_wall,dim=('time','x','z'))
        
        #mean_fluc=xr.DataArray.mean(theta_wall_fluc,dim=('time','x','z'))
        #rms_fluc=((theta_wall_fluc-mean_fluc)**2)
        #rms_fluc=xr.ufuncs.sqrt(xr.DataArray.mean(rms_fluc,dim=('time','x','z')))
        #fluc_over_rms_fluc=theta_wall_fluc/rms_fluc

        #PDF_data[val+'_flux_fluc_over_rmsFluc']=fluc_over_rms_fluc.values.flatten()
        PDF_data[val+'_flux_fluc_over_rmsFluc']=theta_wall.values.flatten()

    k=k+1
    print("done with " +val,flush=True)

#if os.path.exists(save_spot):
    #PDF_data.to_parquet(os.path.join(save_spot,'PDF_data_input_output.parquet'),engine='fastparquet',compression='GZIP')

print("data saved, getting ready to plot..",flush=True)
# %%
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
cm = 1/2.54

fig = plt.figure(figsize=(18*cm,8*cm),dpi=500)
plt.axvline(x = 15, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 30, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 50, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 75, color = 'darkgrey',linewidth=0.8)
plt.plot(r_list["r_pr0.025"],label="pr0.025",color='royalblue')
plt.plot(r_list["r_pr0.2"],label="pr0.2",color='green')
plt.plot(r_list["r_pr0.71"],label="pr0.71",color='darkorange')
plt.plot(r_list["r_pr1"],label="pr1",color='maroon')

plt.grid()
plt.xticks(np.arange(0,105,5))
plt.yticks(np.arange(-1,0.1,0.1))
plt.xscale("log")
plt.xlabel(r'$y^+$',fontsize=15)
#plt.ylabel(r'$\dfrac{u q_w}{u_{rms} {q_w}_{rms}}$')
#plt.ylabel(r'$\dfrac{\overline{u^\prime q_w^\prime}}{u^\prime_{_{RMS}} q^\prime_{w_{RMS}}}$')
plt.ylabel(r'$\overline{v^\prime q_w^\prime} \ \left/ \ v^\prime_{_{RMS}} q^\prime_{w_{RMS}} \right.$',fontsize=12)
plt.legend(loc='lower left')
plt.savefig("/home/au567859/DataHandling/reports/figures/pearsons_uv_qw.png")



#%% Theta
Re_Tau = 395  # Direct from simulation
Re = 10400  # Direct from simulation
nu = 1 / Re  # Kinematic viscosity
u_tau = Re_Tau * nu  # The friction velocity
Q_avg=0.665
val_list=sorted(list(ds.keys()))
#val_list.remove("w_vel")
val_list.remove("u_vel")
#val_list.remove("v_vel")

r_list=pd.DataFrame(columns=['pr0.025','pr0.2','pr0.71','pr1'])
p_list=pd.DataFrame(columns=['pr0.025','pr0.2','pr0.71','pr1'])
PDF_data=pd.DataFrame()
theta_wall_list=pd.DataFrame(columns=['pr0.025','pr0.2','pr0.71','pr1'])

y_100=ds.sel(y=utility.y_plus_to_y(100),method="nearest")["y"].values


print("data loaded, ready to process..",flush=True)
for val in val_list:
    theta_wall_list[val] = (nu/(float(val[2:]))*(ds[val].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))).values.flatten()
    for i in range(2,49):
        data=ds.isel(y=-i)
        y=data["y"].values
        data=data[val].values.flatten()

        # This should be done outside of loop, waste of computing
        theta_wall = (nu/(float(val[2:]))*(ds[val].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest"))).values.flatten()
    
        r, p = stats.pearsonr(data, theta_wall_list[val])
        r_list.loc[(utility.y_to_y_plus(y),val)]=r
        p_list.loc[(utility.y_to_y_plus(y),val)]=p
        
        print("done with y_plus="+str(utility.y_to_y_plus(y)),flush=True)
    print("done with " +val,flush=True)

#if os.path.exists(save_spot):
    #PDF_data.to_parquet(os.path.join(save_spot,'PDF_data_input_output.parquet'),engine='fastparquet',compression='GZIP')

print("data saved, getting ready to plot..",flush=True)

#%%
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
cm = 1/2.54

fig = plt.figure(figsize=(18*cm,8*cm),dpi=500)
plt.axvline(x = 15, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 30, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 50, color = 'darkgrey',linewidth=0.8)
plt.axvline(x = 75, color = 'darkgrey',linewidth=0.8)
plt.plot(r_list["pr0.025"],label="pr0.025",color='royalblue')
plt.plot(r_list["pr0.2"],label="pr0.2",color='green')
plt.plot(r_list["pr0.71"],label="pr0.71",color='darkorange')
plt.plot(r_list["pr1"],label="pr1",color='maroon')

plt.grid()
plt.xticks(np.arange(0,105,5))
plt.yticks(np.arange(0,1.1,0.1))
plt.xscale("log")
plt.xlabel(r'$y^+$',fontsize=15)
plt.ylabel(r'$\overline{\theta^\prime q_w^\prime} \ \left/ \ \theta^\prime_{_{RMS}} q^\prime_{w_{RMS}} \right.$',fontsize=12)
plt.legend(loc='lower left')
plt.savefig("/home/au567859/DataHandling/reports/figures/pearsons_T_qw.png")


# %%
