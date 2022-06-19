#%%
import xarray as xr
from DataHandling import utility
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import os

slice_array=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")

#%% 
Re_Tau = 395
Re = 10400          #Direct from simulation
nu = 1/Re           #Kinematic viscosity
u_tau = Re_Tau*nu   
Q_avg=0.665

tau_wall=slice_array['u_vel'].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
tau_wall=(nu*tau_wall)/u_tau**2
tau_wall = tau_wall.to_numpy()
tau_wall = tau_wall.flatten()

target=['pr0.71_flux']
q_wall=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
pr_number=float(target[0][2:-5])
q_wall=(nu/(pr_number)*q_wall)/Q_avg
q_wall = q_wall.to_numpy()
q_wall = q_wall.flatten()

# # %%
# num_bins = 100
# bin_width = (tau_wall.max() - tau_wall.min()) / num_bins
# hist_area = len(tau_wall) * bin_width

# fig, ax = plt.subplots(dpi=500)
# sns.histplot(tau_wall,bins=num_bins, ax=ax)
# ax.set_xlabel(r'$\tau_{w,x}^+ $')
# ax.set_yticks(np.linspace(0, ax.get_ybound()[1], 8))

# ax2 = ax.twinx()
# sns.kdeplot(tau_wall, ax=ax2,color='orange',linewidth=2)
# ax2.set_xlim(tau_wall.min(),tau_wall.max())
# ax2.set_ylim(ymax=ax.get_ylim()[1] / hist_area)
# ax2.set_yticks(np.linspace(0, ax2.get_ybound()[1], 8))
# fig.savefig(os.path.join("/home/au567859/DataHandling/reports/figures/",'tau_wall_kde.pdf'),bbox_inches='tight',format='pdf')
# plt.clf()

# #%%
# num_bins = 100
# bin_width = (q_wall.max() - q_wall.min()) / num_bins
# hist_area = len(q_wall) * bin_width

# fig2, axs = plt.subplots(dpi=500)
# sns.histplot(q_wall,bins=num_bins, ax=axs)
# axs.set_xlabel(r'$\frac{q_w}{\overline{q_{w,DNS}}},\quad Pr=0.71$')
# axs.set_yticks(np.linspace(0, axs.get_ybound()[1], 8))

# axs2 = axs.twinx()
# sns.kdeplot(q_wall, ax=axs2,color='orange',linewidth=2)
# axs2.set_xlim(q_wall.min(),q_wall.max())
# axs2.set_ylim(ymax=axs.get_ylim()[1] / hist_area)
# axs2.set_yticks(np.linspace(0, axs2.get_ybound()[1], 8))
# fig2.savefig(os.path.join("/home/au567859/DataHandling/reports/figures/",'q_wall_kde.pdf'),bbox_inches='tight',format='pdf')
# plt.clf()

#%%
import seaborn as sns
import matplotlib.pyplot as plt
sns.jointplot(x = tau_wall[0:100], y = q_wall[0:100],
              kind = "kde")
# Show the plot
plt.show()
# fig.savefig(os.path.join("/home/au567859/DataHandling/reports/figures/",'jointpdf.pdf'),bbox_inches='tight',format='pdf')
# plt.clf()
# %%
