
#%%
import os
import numpy as np
from DataHandling import utility
import pandas as pd
import matplotlib.pyplot as plt


path_of_output="/home/au567859/DataHandling/models/output"
name_list, _ = utility.get_runs_wandb()

model=name_list[4]

full_dir=os.path.join(path_of_output,model)
subdirs=os.listdir(full_dir)

print('This is model ' + model,flush=True)

subdirs = subdirs[0]
dir_split = subdirs.split("-")

y_plus = int(dir_split[0][-2:])

index_vars_s = dir_split.index("VARS")
index_target = dir_split.index("TARGETS")

var = dir_split[index_vars_s+1:index_target]
target = dir_split[index_target+1:]
if "normalized" not in dir_split:
    normalize = False
else:
    normalize = True
    target = target[:-1]

if target[0][-5:] == '_flux':
    target_type = "flux"
elif target[0] == 'tau_wall':
    target_type = "stress"

model_path, output_path =utility.model_output_paths(model,y_plus,var,target,normalize)

#%%
train=pd.read_parquet(os.path.join(output_path,'Error_fluct_train.parquet'),engine='fastparquet')
val=pd.read_parquet(os.path.join(output_path,'Error_fluct_validation.parquet'),engine='fastparquet')
test=pd.read_parquet(os.path.join(output_path,'Error_fluct_test.parquet'),engine='fastparquet')

# %%
"""3d KDE of the errors
Args:
error_val (numpy array): the errors
output_path (Path): where to save
Returns:
None: 
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

#change the scale to plus units
Re_Tau = 395 #Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu

error_val=test
train_numpy=error_val.to_numpy()
num_snapshots=int(train_numpy.shape[0]/256/256)
reshape_t=train_numpy.reshape((num_snapshots,256,256))
avg=np.mean(reshape_t,0)

cm = 1/2.54  # centimeters in inches

fig, axs=plt.subplots(1,1,figsize=([21*cm,10*cm]),sharex=True,sharey=True,constrained_layout=False,dpi=1000)

name="fallen-cloud-32"

avg_mean =avg.mean()

for i in range(len(avg)):
    for j in range(len(avg)):
        if avg[i,j]<1.10*avg_mean:
            avg[i,j]=0

pcm=axs.imshow(np.transpose(avg),cmap='viridis',aspect=0.5)
axs.set_title(name.capitalize(),weight="bold")
plt.axis('off')