#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from DataHandling import utility 
import os
import tensorflow as tf
import keras
from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils
from keras.models import Model

#%%
input_features=['u_vel','v_vel','w_vel','pr_mix_0025_02_071_1']
padding_layers=8

input_list=[]
reshape_list=[]
weights=[128,256,256]

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

for features in input_features:
    input=keras.layers.Input(shape=(256,256),name=features)
    input_list.append(input)
    pad=PeriodicPadding2D(padding=padding_layers)(input)
    reshape=keras.layers.Reshape((256+2*padding_layers,256+2*padding_layers,1))(pad)
    reshape_list.append(reshape)

#%%


model_name_moe_30 = 'worldly-hill-105'
model_path_moe_30=os.path.join("/home/au567859/DataHandling/models/trained/",model_name_moe_30)

load_pretrained_model_moe_30 = tf.keras.models.load_model(model_path_moe_30)
pretrained_model_moe_30 = Model(load_pretrained_model_moe_30.input,load_pretrained_model_moe_30.layers[-1].output)
pretrained_model_moe_30.trainable=True

MoE_30_output=pretrained_model_moe_30(input_list,training=True)

model = keras.Model(input_list, outputs=MoE_30_output)
model.summary()

# %%
from DataHandling.features import slices
import xarray as xr 
var=['u_vel','v_vel','w_vel','pr_mix_0025_02_071_1']
target=['pr0.025_flux_mix_pr0.2_pr0.71_pr1']
normalized=False
y_plus=30

#df=xr.open_zarr("/home/au567859/DataHandling/data/interim/data.zarr")
#slices.save_tf(y_plus,var,target,df,normalized=normalized)

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

data=slices.load_from_scratch(y_plus,var,target,normalized,repeat=repeat,shuffle_size=shuffle,batch_s=batch_size)
train=data[0]
validation=data[1]
#%%
import wandb

wandb.init(project="Thesis",notes="Mixture of Experts Linear pr0025&pr071")



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
