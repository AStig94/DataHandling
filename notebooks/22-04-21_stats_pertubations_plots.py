#%% Checking for convergence
from DataHandling import plots
fn = '/home/au567859/DataHandling/data_test/stats.nc'
batch_size = 3
pr_list = ['pr0.69', 'pr0.22', 'pr0.18', 'pr0.045']
plots.stat_plots(fn,batch_size,pr_list)

# %%
