#%%

from DataHandling import utility 
from DataHandling.models import predict
import pandas as pd
import re
import os
import numpy as np


name_list, config_list = utility.get_runs_wandb()

y_plus_list = []
pr_list = []
E_rms_list = []

normalized=False


#(u,v,w)
models=['winter-firefly-75','unique-night-53','exalted-durian-66','pleasant-snowflake-76','amber-glitter-73','worthy-bush-58','frosty-lion-63','super-pond-78','different-silence-74','devoted-bee-61','glad-oath-62','solar-water-77','clear-donkey-82','earthy-universe-83','effortless-totem-85','deep-terrain-84']

#(u,v,w,T)
#models=['youthful-wave-72','bumbling-shape-52','zany-snowball-67','flowing-planet-79','silvery-dragon-71','glowing-sponge-59','crisp-galaxy-64','rich-moon-81','cool-bird-51','treasured-music-60','skilled-planet-65','driven-gorge-80','true-salad-89','iconic-monkey-87','stellar-bee-88','divine-glitter-86']

for i in models:
    var = config_list[name_list.index(i)]['variables']
    y_plus = config_list[name_list.index(i)]['y_plus']
    var = config_list[name_list.index(i)]['variables']
    target=[config_list[name_list.index(i)]['target']]
    pr = float(re.findall(r"[-+]?\d*\.\d+|\d+", target[0])[0])

    model_path, output_path = utility.model_output_paths(i,y_plus,var,target,normalized)
    csv_file = os.path.join(output_path, "Mean_error.csv")
    my_data = pd.read_csv(csv_file)
    E_rms = my_data['Root mean sq. error of local heat flux'][2]
    
    y_plus_list = np.append(y_plus_list,y_plus)
    pr_list = np.append(pr_list,pr)
    E_rms_list = np.append(E_rms_list,E_rms)



#%%
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

npts = np.size(E_rms_list)
x = pr_list
y = y_plus_list
z = E_rms_list

y_plus_list_scaled = []
for i in range(len(y_plus_list)):
    y_plus_scaled=y_plus_list[i]*pr_list[i]**(0.5)
    y_plus_list_scaled=np.append(y_plus_list_scaled,y_plus_scaled)
fig, ax = plt.subplots(dpi=500)

#ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
ax.tricontour(x, y_plus_list_scaled, z, levels=14, linewidths=0.5, colors='k')
#cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
cntr = ax.tricontourf(x, y_plus_list_scaled, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax,label=r'$E_{RMS}$ [%]')
#x.plot(x, y, 'ko', ms=3)
ax.plot(x, y_plus_list_scaled, 'ko', ms=3)
#ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax.set(xlim=(x.min(), x.max()), ylim=(y_plus_list_scaled.min(), y_plus_list_scaled.max()))
#ax.set_title(r'tricontour (%d points) - $(u,v,w,\theta)$' % npts)
#ax.set_ylabel(r'$y^{+}$')
ax.set_ylabel(r'$y^{+}_{scaled}$')
ax.set_xlabel('Prandtl number')


plt.show()
