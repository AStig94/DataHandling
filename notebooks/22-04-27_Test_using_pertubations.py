
#%%
def error(target_list,target_type,names,predctions,output_path):
    
    

    import os
    import numpy as np
    import pandas as pd
    from numba import njit

    @njit(cache=True,parallel=True)    
    def cal_func(target_list,predctions):
        
        fluc_predict=predctions-np.mean(predctions)
        fluc_target=target_list-np.mean(target_list)
        

        #Global average errors
        global_mean_err=(np.mean(predctions)-np.mean(target_list))/(np.mean(target_list))*100
        MSE_local_shear_stress=np.sqrt((np.mean((predctions-target_list)**2))/np.mean(target_list))*100
        MSE_local_fluct_shear_stress=np.sqrt((np.mean((fluc_predict-fluc_target)**2))/np.mean(fluc_target))*100
        global_fluct_error=(np.std(fluc_predict)-np.std(fluc_target))/(np.std(fluc_target))*100
        MSE_local_fluc=np.sqrt((np.mean((fluc_predict-fluc_target)**2))/np.std(fluc_target)**2)*100


        

        #MAE_local=np.mean(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_local_no_mean=(np.abs(predctions[i][:,:,:]-target_list[i][:,:,:]))/np.mean(np.abs(target_list[i][:,:,:]))*100
        #MAE_fluct_no_mean=(np.abs(fluc_predict-fluc_target))/np.mean(np.abs(fluc_target))*100
        

        

        #Local erros for PDF's and boxplots etc.
        MSE_local_no_mean=np.sqrt(((predctions-target_list)**2)/np.mean(target_list)**2)*100
        #MSE_local_fluc_PDF=np.sqrt(((fluc_predict-fluc_target)**2)/(np.std(fluc_target))**2)*100
        
        return MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc,MSE_local_fluct_shear_stress


    if not os.path.exists(output_path):
        os.makedirs(output_path)

 
    
    if target_type=="stress":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local shear stress','Global fluctuations error','Root mean sq. error of fluctuating local shear stress'])
    elif target_type=="flux":
        error=pd.DataFrame(columns=['Global Mean Error','Root mean sq. error of local heat flux','Global fluctuations error','Root mean sq. error of local fluctuations','Root mean sq. error of fluctuating local shear stress'])
    
    error_fluc_list=[]
    
    


    
    for i in range(3):
        error_fluct=pd.DataFrame()
        
        MSE_local_no_mean,global_mean_err,MSE_local_shear_stress,global_fluct_error,MSE_local_fluc,MSE_local_fluct_shear_stress=cal_func(target_list[i],predctions[i])


        if target_type=="stress":
            error_fluct['Root sq. error of local shear stress']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local shear stress':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc,'Root mean sq. error of fluctuating local shear stress':MSE_local_fluct_shear_stress},ignore_index=True)
        elif target_type=="flux":
            error_fluct['Root sq. error of local heat flux']=MSE_local_no_mean.flatten()
            error=error.append({'Global Mean Error':global_mean_err,'Root mean sq. error of local heat flux':MSE_local_shear_stress,'Global fluctuations error':global_fluct_error,'Root mean sq. error of local fluctuations':MSE_local_fluc,'Root mean sq. error of fluctuating local shear stress':MSE_local_fluct_shear_stress},ignore_index=True)
        
        #error_fluct['Root sq. error of local fluctuations']=MSE_local_fluc_PDF.flatten()
        #error_fluct['MAE local']=MAE_local_no_mean.flatten()
        #error_fluct['MAE fluct']=MAE_fluct_no_mean.flatten()

        

        error_fluct.to_parquet(os.path.join(output_path,'Error_fluct_'+names[i]+'.parquet'),engine='fastparquet',compression='GZIP')
        error_fluc_list.append(error_fluct)
        

    
    error.index=names

    #error.to_csv(os.path.join(output_path,'Mean_error.csv'))

    return error_fluc_list, error

#%% Import MoE 
from DataHandling import utility 
from DataHandling.models import predict
import keras
import os

name_list, config_list = utility.get_runs_wandb()
model_name='treasured-music-60'
y_plus = config_list[name_list.index(model_name)]['y_plus']
var = config_list[name_list.index(model_name)]['variables']
normalized = config_list[name_list.index(model_name)]['normalized']
target=[config_list[name_list.index(model_name)]['target']]

model_path, _ = utility.model_output_paths(model_name,y_plus,var,target,normalized)
if os.path.exists(model_path):
    model=keras.models.load_model(model_path)

#%%
var=['u_vel','v_vel','w_vel','pr0.69']
target=['pr0.69_flux']
normalized=False
overwrite=False
y_plus=30

for layer in model.layers:
    #if layer._name=='pr_mix_0025_02_071_1':
    if layer._name=='pr0.71':
        layer._name='pr0.69'

#%%
import xarray as xr
from DataHandling.features import slices

#df=xr.open_zarr("/home/au567859/DataHandling/data_test/interim/data.zarr")
#slices.save_tf(y_plus,var,target,df,test=True)

#%%
predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=True)

# %%
import shutil
import numpy as np
from zipfile import BadZipfile
import os
from DataHandling import utility


path_of_output="/home/au567859/DataHandling/models/output"
full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)
dir=subdirs[0]

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

model_path, output_path =utility.model_output_paths(model,y_plus,var,target,normalize,test=True)

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

error_fluc_list, error2 = error(target_list,target_type,names,predctions,output_path)
# %%
