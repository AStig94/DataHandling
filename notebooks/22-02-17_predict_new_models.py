#%% Predict new models
from DataHandling import utility 
from DataHandling.models import predict
import keras
import wandb
import os
import tensorflow as tf


name_list, config_list = utility.get_runs_wandb()
overwrite=False
normalized=False


for i in name_list:
    y_plus = config_list[name_list.index(i)]['y_plus']
    var = config_list[name_list.index(i)]['variables']
    if type(config_list[name_list.index(i)]['target'])==list:
        target=config_list[name_list.index(i)]['target']
    else:
        target=[config_list[name_list.index(i)]['target']]
    model_path, output_path = utility.model_output_paths(i,y_plus,var,target,normalized)
    if os.path.exists(model_path):
        model=keras.models.load_model(model_path)
        predict.predict(i,overwrite,model,y_plus,var,target,normalized)


# %%
