

#%%
import dask
import xarray as xr
from DataHandling.features import stats
from DataHandling import utility

client,cluster=utility.slurm_q64(1)
data=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")


data=data.isel(time=slice(0,50))



stats.calc_stats(data,"/home/au567859/DataHandling/data/test.nc")