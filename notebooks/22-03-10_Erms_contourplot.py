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

for i in name_list:
    var = config_list[name_list.index(i)]['variables']
    if 'padding' in config_list[name_list.index(i)] and np.size(var)==3:
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
    else:  
        pass


#%%
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

npts = np.size(E_rms_list)
x = pr_list
y = y_plus_list
z = E_rms_list

fig, ax = plt.subplots(dpi=500)

ax.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")

fig.colorbar(cntr, ax=ax,label=r'$E_{RMS}$ [%]')
ax.plot(x, y, 'ko', ms=3)
ax.set(xlim=(x.min(), x.max()), ylim=(y.min(), y.max()))
ax.set_title(r'tricontour (%d points) - $(u,v,w)$' % npts)
ax.set_ylabel(r'$y^{+}$')
ax.set_xlabel('Prandtl number')


plt.show()
