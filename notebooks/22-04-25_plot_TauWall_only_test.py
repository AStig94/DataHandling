#%%
""" Det her script kører igemmen alle modeller fra Wandb af i SlURM array mode.
    Ud fra modelnavnet finder den frem til hvor modellen er gemt og hvor dens data ligger i models/output. Hvis der ikke findes en paraquet fil med error
    data generes dette, og ellers springer den over(vedmindre overwrite er sat til true). Derefter generes billeder
"""

import os
import matplotlib
import importlib

#Fjern den her hvis der skal køres interaktivt. Skal bruges for at kunne lave billeder i batch mode
matplotlib.use('Agg')


import numpy as np
from glob import glob
from DataHandling import utility
from DataHandling import plots
from zipfile import BadZipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt



overwrite=False
overwrite_pics=True
overwrite_pdf=True

path_of_output="/home/au567859/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

#slurm_arrary_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))



#%%
#model=name_list[slurm_arrary_id]
#importlib.reload(plots)
model='skilled-planet-65'


full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)

print('This is model ' + model,flush=True)

for dir in subdirs:
    if 'mix' in dir:
        pass
    else:
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
                #if os.path.exists(os.path.join(output_path,"test_PDF.png")):
                plots.heatmap_quarter_test(predctions[2],target_list[2],output_path,target)

            plt.close('all')
            print("done with " +model,flush=True)


#%%

#output_path
#importlib.reload(plots)
#plots.heatmap_quarter_test(predctions[0],target_list[0],output_path,target)


