# %%
import pandas as pd
import xarray as xr
from DataHandling import utility
import seaborn as sns
import numpy as np
import os
import matplotlib
from random import sample
from zipfile import BadZipfile
import shutil
matplotlib.use('Agg')
Q_avg=0.665
Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

path_of_output="/home/au567859/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

models=['wild-voice-174','solar-snowflake-175','winter-dust-176','silver-planet-177','youthful-wave-72','bumbling-shape-52','zany-snowball-67','flowing-planet-79','silvery-dragon-71','glowing-sponge-59','crisp-galaxy-64','rich-moon-81','cool-bird-51','treasured-music-60','skilled-planet-65','driven-gorge-80','true-salad-89','iconic-monkey-87','stellar-bee-88','divine-glitter-86']
pred_models_stress=pd.DataFrame()
models_stress=[]
pred_models_flux=pd.DataFrame()
models_flux=[]


for model in models:
    full_dir=os.path.join(path_of_output,model)
    subdirs=os.listdir(full_dir)
    print('This is model ' + model,flush=True)
    for dir in subdirs:
        dir_split = dir.split("-")

        y_plus = int(dir_split[0][-2:])

        index_vars_s = dir_split.index("VARS")
        index_target = dir_split.index("TARGETS")

        var = dir_split[index_vars_s+1:index_target]

        if '|' in dir_split[index_target+1:][0]:
            target=dir_split[index_target+1:][0].split('|')
        else:
            target = dir_split[index_target+1:] 
            
        if "normalized" not in dir_split:
            normalize = False
        else:
            normalize = True
            target = target[:-1]

        if 'flux' in target[0]:
            target_type = "flux"
        elif 'tau_wall' in target[0]:
            target_type = "stress"

        model_path, output_path =utility.model_output_paths(model,y_plus,var,target,normalize)

        prediction_path=os.path.join(output_path,'predictions.npz')
        target_path=os.path.join(output_path,'targets.npz')

        if os.path.exists(prediction_path) and os.path.exists(target_path):
            try:
                scratch=os.path.join('/scratch/', os.environ['SLURM_JOB_ID'])
                prediction_scratch=os.path.join(scratch,'predictions.npz')
                target_scratch=os.path.join(scratch,'targets.npz')
                shutil.copy2(prediction_path,prediction_scratch)
                shutil.copy2(target_path,target_scratch)
                
                pred=np.load(prediction_scratch)
                targ=np.load(target_scratch)
            except BadZipfile:
                print("Npz file is corroupt, make new")
                shutil.rmtree(output_path)
            else:
                target_list=[targ["train"],targ["val"],targ["test"]]
                predctions=[pred["train"],pred["val"],pred["test"]]
                names=["train","validation","test"]

                if 'tau_wall' in target[0]:
                    models_stress+=[model]
                    data=(nu*predctions[2]/u_tau**2).flatten()
                    fluc=data-np.mean(data)
                    rms_fluc=np.sqrt(np.mean((fluc-np.mean(fluc))**2))
                    pred_models_stress[model]=fluc/rms_fluc

                else:
                    models_flux+=[model]
                    data=(predctions[2]/Q_avg).flatten()
                    fluc=data-np.mean(data)
                    rms_fluc=np.sqrt(np.mean((fluc-np.mean(fluc))**2))
                    pred_models_flux[model]=fluc/rms_fluc

# %%
import matplotlib.pyplot as plt

cm = 1/2.54  # centimeters in inches
fig, axs=plt.subplots(nrows=4,ncols=4,figsize=([1.4*21*cm,0.7*21*cm]),dpi=1000,sharex=True,sharey=True,constrained_layout=False)

for model in models_flux:
    i = models_flux.index(model)
    row, col = divmod(i,4)
    flux=pred_models_flux[model]
    stress=pred_models_stress[models_stress[row]]
    xmin = flux.mean()-3*flux.std()
    xmax = flux.mean()+3*flux.std()
    ymin = stress.mean()-3*stress.std()
    ymax = stress.mean()+3*stress.std()
    sns.jointplot(ax=axs[row,col],x=flux, y=stress, kind="kde",xlim=(xmin,xmax),ylim=(ymin,ymax))
plt.savefig("PDF.png",dpi=500)

# #%%

# Re = 10400 #Direct from simulation
# nu = 1/Re #Kinematic viscosity
# save_spot=os.path.join("/home/au567859/DataHandling/data/")

# ds=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")

# sequence = [i for i in range(ds.sizes["time"])]
# subset = sample(sequence, 2)

# ds=ds.isel(time=subset)
# y_plus=[75]
# #%%

# Re_Tau = 395  # Direct from simulation
# Re = 10400  # Direct from simulation
# nu = 1 / Re  # Kinematic viscosity
# u_tau = Re_Tau * nu  # The friction velocity
# Q_avg=0.665
# val_list=sorted(list(ds.keys()))
# # mean=ds
# # mean=mean.drop(labels=val_list)

# PDF_data=pd.DataFrame()

# print("data loaded, ready to process..",flush=True)
# k=0
# for val in val_list:
#     if k>=4:
#         for i in y_plus:
#             data=ds.sel(y=utility.y_plus_to_y(i),method="nearest")
#             mean_val=xr.DataArray.mean(data[val],dim=('time','x','z'))
            
#             fluc = data[val]-mean_val
#             mean_fluc=xr.DataArray.mean(fluc,dim=('time','x','z')) # Should be zero

#             rms_fluc=((fluc-mean_fluc)**2)
#             rms_fluc=xr.ufuncs.sqrt(xr.DataArray.mean(rms_fluc,dim=('time','x','z')))

#             fluc_over_rms_fluc=fluc/rms_fluc
#             PDF_data[val+'_fluc_over_rmsFluc_yplus_'+str(i)]=fluc_over_rms_fluc.values.flatten()
#             print("done with y_plus=" +str(i),flush=True)
#     else:
#         theta_wall = (nu/(float(val[2:]))*(ds[val].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")))
#         theta_wall_fluc = theta_wall - xr.DataArray.mean(theta_wall,dim=('time','x','z'))
        
#         mean_fluc=xr.DataArray.mean(theta_wall_fluc,dim=('time','x','z'))
#         rms_fluc=((theta_wall_fluc-mean_fluc)**2)
#         rms_fluc=xr.ufuncs.sqrt(xr.DataArray.mean(rms_fluc,dim=('time','x','z')))
#         fluc_over_rms_fluc=theta_wall_fluc/rms_fluc

#         # theta_wall = ds[val].sel(y=utility.y_plus_to_y(15),method="nearest")
#         # theta_wall_fluc = theta_wall - xr.DataArray.mean(theta_wall,dim=('time','x','z'))
        
#         # mean_fluc=xr.DataArray.mean(theta_wall_fluc,dim=('time','x','z'))
#         # rms_fluc=((theta_wall_fluc-mean_fluc)**2)
#         # rms_fluc=xr.ufuncs.sqrt(xr.DataArray.mean(rms_fluc,dim=('time','x','z')))
#         # fluc_over_rms_fluc=theta_wall_fluc/rms_fluc

#         PDF_data[val+'_flux_fluc_over_rmsFluc']=fluc_over_rms_fluc.values.flatten()
#         # maybe this should also be fluc

#     k=k+1
#     print("done with " +val,flush=True)

# if os.path.exists(save_spot):
#     PDF_data.to_parquet(os.path.join(save_spot,'PDF_data_input_output.parquet'),engine='fastparquet',compression='GZIP')

# print("data saved, getting ready to plot..",flush=True)
# # %%
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import seaborn as sns
# save_spot=os.path.join("/home/au567859/DataHandling/data/")
# dataset=pd.read_parquet(os.path.join(save_spot,'PDF_data_input_output.parquet'),engine='fastparquet')
# cm =1/2.54

# #%%
# import scipy.stats as stats
# dataset=dataset[0:20000]
# #dataset=pd.read_parquet(os.path.join(save_spot,'PDF_data_input_output.parquet'),engine='fastparquet')

# x="u_vel_fluc_over_rmsFluc_yplus_75"
# y="pr0.025_flux_fluc_over_rmsFluc"

# xmin =dataset[x].mean()-3*dataset[x].std()
# xmax = dataset[x].mean()+3*dataset[x].std()
# ymin = dataset[y].mean()-3*dataset[y].std()
# ymax = dataset[y].mean()+3*dataset[y].std()
# plot = sns.jointplot(dataset[x], y=dataset[y], kind="kde",xlim=(xmin,xmax),ylim=(ymin,ymax))

# r, p = stats.pearsonr(dataset[x], dataset[y])
# phantom, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
# # here graph is not a ax but a joint grid, so we access the axis through ax_joint method
# plot.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])
# plot.set_axis_labels(xlabel=r'$\dfrac{u^\prime}{u^\prime_{rms}}$',ylabel=r'$\dfrac{{q_w}^\prime}{{q_w}^\prime_{rms}}$')
# plot.savefig("out.png",dpi=500)