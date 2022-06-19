#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DataHandling import utility 
import xarray as xr
import os
from DataHandling.features import slices

#%%
name_list, config_list = utility.get_runs_wandb()

model_name = 'usual-armadillo-133'
model_path=os.path.join("/home/au567859/DataHandling/models/trained/",model_name)
#%%
from keras.models import Model
load_pretrained_model = tf.keras.models.load_model(model_path)
pretrained_model = Model(load_pretrained_model.input,load_pretrained_model.layers[-2].output)
pretrained_model.trainable=False

#%%
import tensorflow as tf
import keras
from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils

padding_layers=8
activation='elu'

x=pretrained_model(pretrained_model.inputs,training=False)

class PeriodicPadding2D(keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        super(PeriodicPadding2D, self).__init__(**kwargs)
        self.padding = conv_utils.normalize_tuple(padding, 1, 'padding')
        self.input_spec = InputSpec(ndim=3)

    def wrap_pad(self, input, size):
        M1 = tf.concat([input[:,:, -size:], input, input[:,:, 0:size]], 2)
        M1 = tf.concat([M1[:,-size:, :], M1, M1[:,0:size, :]], 1)
        return M1

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  
        if shape[1] is not None:
            length = shape[1] + 2*self.padding[0]
        else:
            length = None
        return tuple([shape[0], length, length])

    def call(self, inputs): 
        return self.wrap_pad(inputs, self.padding[0])

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(PeriodicPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


weights=[128,256,256]


batch1=keras.layers.BatchNormalization(-1)(x)
cnn1=keras.layers.Conv2D(64,5,activation=activation)(batch1)

batch2=keras.layers.BatchNormalization(-1)(cnn1)
cnn2=keras.layers.Conv2D(weights[0],3,activation=activation)(batch2)
batch3=keras.layers.BatchNormalization(-1)(cnn2)
cnn3=keras.layers.Conv2D(weights[1],3,activation=activation)(batch3)
batch4=keras.layers.BatchNormalization(-1)(cnn3)
cnn4=keras.layers.Conv2D(weights[2],3,activation=activation)(batch4)
batch5=keras.layers.BatchNormalization(-1)(cnn4)

conc1=keras.layers.Concatenate()([cnn4,batch5])
cnn5=keras.layers.Conv2DTranspose(weights[0],3,activation=activation)(conc1)
batch6=keras.layers.BatchNormalization(-1)(cnn5)

conc2=keras.layers.Concatenate()([cnn3,batch6])
cnn6=keras.layers.Conv2DTranspose(weights[1],3,activation=activation)(conc2)
batch7=keras.layers.BatchNormalization(-1)(cnn6)

conc3=keras.layers.Concatenate()([cnn2,batch7])
cnn7=keras.layers.Conv2DTranspose(weights[2],3,activation=activation)(conc3)
batch8=keras.layers.BatchNormalization(-1)(cnn7)

conc4=keras.layers.Concatenate()([cnn1,batch8])
cnn8=tf.keras.layers.Conv2DTranspose(64,5,activation=activation)(conc4)
batch9=keras.layers.BatchNormalization(-1)(cnn8)

conc5=keras.layers.Concatenate()([x,batch9])
output=tf.keras.layers.Conv2DTranspose(1,1)(conc5)
output=keras.layers.Cropping2D(cropping=padding_layers)(output)

model = keras.Model(pretrained_model.inputs, outputs=output)
model.summary()
#%%
from DataHandling.features import slices
var=['u_vel',"v_vel","w_vel","pr0.71"]
target=['pr0.71_flux']
normalized=False
y_plus=30


#df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
#slices.save_tf(y_plus,var,target,df,normalized=normalized)

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

data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]
#%%
import wandb

wandb.init(project="Thesis",notes="y_plus_30 wall heat flux transfer learning with IMD model at y_plus_15")



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
config.padding=padding_layers


model.compile(loss=loss, optimizer=optimizer)


#%%
from wandb.keras import WandbCallback
logdir, backupdir= utility.get_run_dir(wandb.run.name)



backup_cb=tf.keras.callbacks.ModelCheckpoint(os.path.join(backupdir,'weights.{epoch:02d}'),save_best_only=False)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=patience,
restore_best_weights=True)
model.fit(x=train,epochs=100000,validation_data=validation,callbacks=[WandbCallback(),early_stopping_cb,backup_cb])

model.save(os.path.join("/home/au567859/DataHandling/models/trained",wandb.run.name))


