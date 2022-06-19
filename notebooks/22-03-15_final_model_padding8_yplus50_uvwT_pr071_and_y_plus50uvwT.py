

#%%
import os
from tensorflow import keras
from keras import layers
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from DataHandling.features import slices
from DataHandling import utility
from DataHandling.models import models
import xarray as xr
os.environ['WANDB_DISABLE_CODE']='True'



# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass


var=['u_vel',"v_vel","w_vel","pr0.71"]
target=['pr0.71_flux']
normalized=False
y_plus=50

y_plus_2=30
var_2 = [i +'_'+str(y_plus_2) for i in var]

# slice_array=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
# var.append(target[0])
# Re = 10400 #Direct from simulation
# nu = 1/Re #Kinematic viscosity
# target_slice1=slice_array[target[0][:-5]].differentiate('y').sel(y=utility.y_plus_to_y(0),method="nearest")
# pr_number=float(target[0][2:-5])
# target_slice1=nu/(pr_number)*target_slice1
# wall_1=slice_array.sel(y=utility.y_plus_to_y(y_plus),method="nearest")

# # New input if two y_plus_values
# if var_2 !=None and y_plus_2!=None:
#     wall_2 = slice_array.sel(y=utility.y_plus_to_y(y_plus_2),method="nearest")
#     wall_1[var_2]=wall_2[var.pop]
#     for feature in var_2:
#         var.append(feature)

# wall_1[target[0]]=target_slice1
# wall_1=wall_1[var]  # Remember target is appended to var further up


#df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
#slices.save_tf(y_plus,var,target,df,normalized=normalized,var_second=var_2,y_plus_second=y_plus_2)

#%%
dropout=False
skip=4
model_type="baseline"
repeat=3
shuffle=100
batch_size=10
activation='elu'
optimizer="adam"
loss='mean_squared_error'
patience=50

for feature in var_2:
    var.append(feature)

data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]

padding=8
model=models.final_skip_no_sep_padding(var,activation,padding)
model.summary()

#keras.utils.plot_model(model,show_shapes=True,dpi=100)

#%%

wandb.init(project="Thesis",notes="y_plus_50 and y_plus_30 as inputs")



config=wandb.config
config.y_plus=y_plus
config.repeat=repeat
config.shuffle=shuffle
config.batch_size=batch_size
config.activation=activation
config.optimizer=optimizer
config.loss=loss
config.patience=patience
config.variables=var
config.target=target[0]
config.dropout=dropout
config.normalized=normalized
config.skip=skip
config.model=model_type
config.padding=padding



model.compile(loss=loss, optimizer=optimizer)


#%%

logdir, backupdir= utility.get_run_dir(wandb.run.name)



backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
model.fit(x=train,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

model.save(os.path.join("/home/au567859/DataHandling/models/trained",wandb.run.name))

