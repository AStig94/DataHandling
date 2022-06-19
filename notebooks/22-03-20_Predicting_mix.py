#%% Predict pr0.025
from DataHandling import utility 
from DataHandling.models import predict
import keras
import xarray as xr
from DataHandling.features import slices
var=['u_vel','v_vel','w_vel','pr0.025']
target=['pr0.025_flux']
normalized=False
y_plus=75

#df=xr.open_zarr("/home/au567859/DataHandling/data_test/interim/data.zarr")
#slices.save_tf(y_plus,var,target,df,test_split=0.7,validation_split=0.2,test=True)

name_list, _ = utility.get_runs_wandb()
model_name='clone-republic-186'
model_path, _ = utility.model_output_paths(model_name,y_plus,var,target,normalized)
overwrite=False
normalized=False

model=keras.models.load_model(model_path)

for layer in model.layers:
    if layer._name=='pr_mix_0025_02_071_1':
        layer._name='pr0.025'

predict.predict(model_name,overwrite,model,y_plus,var,target,normalized)
#predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=True)

#%% Predict pr0.2
var=['u_vel','v_vel','w_vel','pr0.2']
target=['pr0.2_flux']

#name_list, _ = utility.get_runs_wandb()
#model_name='vivid-lion-179'
#model_path, _ = utility.model_output_paths(model_name,y_plus,var,target,normalized)

model=keras.models.load_model(model_path)

for layer in model.layers:
    if layer._name=='pr_mix_0025_02_071_1':
        layer._name='pr0.2'

predict.predict(model_name,overwrite,model,y_plus,var,target,normalized)
#predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=True)

#%% Predict pr0.71
var=['u_vel','v_vel','w_vel','pr0.71']
target=['pr0.71_flux']

#name_list, _ = utility.get_runs_wandb()
#model_name='vivid-lion-179'
#model_path, _ = utility.model_output_paths(model_name,y_plus,var,target,normalized)
#overwrite=False
#normalized=False

model=keras.models.load_model(model_path)

for layer in model.layers:
    if layer._name=='pr_mix_0025_02_071_1':
        layer._name='pr0.71'

predict.predict(model_name,overwrite,model,y_plus,var,target,normalized)
#predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=True)

#%% Predict pr1
var=['u_vel','v_vel','w_vel','pr1']
target=['pr1_flux']

# name_list, _ = utility.get_runs_wandb()
# model_name='still-wind-161'
# model_path, _ = utility.model_output_paths(model_name,y_plus,var,target,normalized)
# overwrite=False
# normalized=False

model=keras.models.load_model(model_path)
#predict.predict(model_name,overwrite,model,y_plus,var,target,normalized,test=True)


for layer in model.layers:
    if layer._name=='pr_mix_0025_02_071_1':
        layer._name='pr1'

predict.predict(model_name,overwrite,model,y_plus,var,target,normalized)
