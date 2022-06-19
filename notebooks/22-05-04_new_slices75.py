from DataHandling.features import slices
import xarray as xr 
var=['u_vel','v_vel','w_vel','pr_mix_0025_02_071_1']
target=['pr0.025_flux_mix_pr0.2_pr0.71_pr1']
normalized=False
y_plus=75

df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
slices.save_tf(y_plus,var,target,df,normalized=normalized)
