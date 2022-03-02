#%% Predict new models
from DataHandling import utility 
from DataHandling.models import predict
import keras
import wandb
import os

name_list, config_list = utility.get_runs_wandb()
y_plus=15
overwrite=False
normalized=False

for i in name_list:
    var = config_list[name_list.index(i)]['variables']
    target=[config_list[name_list.index(i)]['target']]
    model_path, output_path = utility.model_output_paths(i,y_plus,var,target,normalized)
    if os.path.exists(model_path):
        model=keras.models.load_model(model_path)
        predict.predict(i,overwrite,model,y_plus,var,target,normalized)

# %%
