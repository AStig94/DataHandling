# %%
"Testing if the scalars are shifted from the velocity fields. "

import xarray as xr
df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
import matplotlib.pyplot as plt

Re_Tau = 395 # Direct from simulation
Re = 10400 #Direct from simulation
nu = 1/Re #Kinematic viscosity
u_tau = Re_Tau*nu
y_at_yplus_15 = 15*nu/u_tau

u_15=df['u_vel'].isel(time=-1).sel(y=y_at_yplus_15,method='nearest')
plt.figure(dpi=250)
u_15.plot(cmap='viridis',add_labels=False)
plt.ylabel('x')
plt.xlabel('z')
plt.title('u_vel - at y_plus = 15')
plt.show()

theta_15=df['pr0.71'].isel(time=-1).sel(y=y_at_yplus_15,method='nearest')
plt.figure(dpi=250)
theta_15.plot(cmap='viridis',add_labels=False)
plt.ylabel('x')
plt.xlabel('z')
plt.title('theta - at y_plus = 15')
plt.show()

# %%
"Plotting v and w"

v_15=df['v_vel'].isel(time=-1).sel(y=y_at_yplus_15,method='nearest')
plt.figure(dpi=250)
v_15.plot(cmap='viridis',add_labels=False)
plt.ylabel('x')
plt.xlabel('z')
plt.title('v_vel - at y_plus = 15')
plt.show()

w_15=df['w_vel'].isel(time=-1).sel(y=y_at_yplus_15,method='nearest')
plt.figure(dpi=250)
w_15.plot(cmap='viridis',add_labels=False)
plt.ylabel('x')
plt.xlabel('z')
plt.title('w_vel - at y_plus = 15')
plt.show()

# %%
