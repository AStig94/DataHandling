

#%%
from DataHandling.data_raw.make_dataset import append_tozarr

store="/home/au567859/DataHandling/data_test/interim/data.zarr"
raw = "/home/au567859/DataHandling/data_test/raw/"
a=append_tozarr(store,raw)

