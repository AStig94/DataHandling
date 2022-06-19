#%%
import xarray as xr
ds = xr.open_zarr('/home/au567859/DataHandling/data_test/interim/data.zarr')
import xarray as xr
import dask
Re_Tau = 395  # Direct from simulation
Re = 10400  # Direct from simulation
nu = 1 / Re  # Kinematic viscosity
u_tau = Re_Tau * nu  # The friction velocity
ds=ds.assign_coords(y=(ds.y*u_tau/nu))
ds=ds.rename({'y':'y_plus'})

# %%
from DataHandling.features import stats
stats.calc_stats(ds,'/home/au567859/DataHandling/data_test/')


