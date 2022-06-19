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
import xarray as xr


#%%
input_features=['u_vel','v_vel','w_vel','pr_mix_0025_02_1']
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
#%% Pretrained y_plus 30 and pr0.025

model_name_pr0025 = 'bumbling-shape-52'
model_path_pr0025=os.path.join("/home/au567859/DataHandling/models/trained/",model_name_pr0025)

load_pretrained_model_pr0025 = tf.keras.models.load_model(model_path_pr0025)
pretrained_model_pr0025 = Model(load_pretrained_model_pr0025.input,load_pretrained_model_pr0025.layers[-2].output)
pretrained_model_pr0025.trainable=False

# for layer in pretrained_model_pr0025.layers:
#     if layer._name=='pr0.025':
#         layer._name='pr_mix_0025_071'

FCN_pr0025_output=pretrained_model_pr0025(input_list,training=False)

#%% Pretrained y_plus 30 and pr0.2

model_name_pr02 = 'glowing-sponge-59'
model_path_pr02=os.path.join("/home/au567859/DataHandling/models/trained/",model_name_pr02)

load_pretrained_model_pr02 = tf.keras.models.load_model(model_path_pr02)
pretrained_model_pr02 = Model(load_pretrained_model_pr02.input,load_pretrained_model_pr02.layers[-2].output)
pretrained_model_pr02.trainable=False

# for layer in pretrained_model_pr0025.layers:
#     if layer._name=='pr0.025':
#         layer._name='pr_mix_0025_071'

FCN_pr02_output=pretrained_model_pr02(input_list,training=False)
#%% Pretrained y_plus 30 and pr0.71

# model_name_pr071 = 'treasured-music-60'
# model_path_pr071=os.path.join("/home/au567859/DataHandling/models/trained/",model_name_pr071)

# load_pretrained_model_pr071 = tf.keras.models.load_model(model_path_pr071)
# pretrained_model_pr071 = Model(load_pretrained_model_pr071.input,load_pretrained_model_pr071.layers[-2].output)
# pretrained_model_pr071.trainable=False

# # for layer in pretrained_model_pr071.layers:
# #     if layer._name=='pr0.71':
# #         layer._name='pr_mix_0025_071'

# FCN_pr071_output=pretrained_model_pr071(input_list,training=False)

#%% Pretrained y_plus 30 and pr1

model_name_pr1 = 'iconic-monkey-87'
model_path_pr1=os.path.join("/home/au567859/DataHandling/models/trained/",model_name_pr1)

load_pretrained_model_pr1 = tf.keras.models.load_model(model_path_pr1)
pretrained_model_pr1 = Model(load_pretrained_model_pr1.input,load_pretrained_model_pr1.layers[-2].output)
pretrained_model_pr1.trainable=False

# for layer in pretrained_model_pr071.layers:
#     if layer._name=='pr0.71':
#         layer._name='pr_mix_0025_071'

FCN_pr1_output=pretrained_model_pr1(input_list,training=False)
#%%
import tensorflow as tf
import keras
from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils

padding_layers=8
activation='elu'

conc=keras.layers.Concatenate()(reshape_list)
batch1=keras.layers.BatchNormalization(-1)(conc)
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

conc5=keras.layers.Concatenate()([conc,batch9])
cnn9=tf.keras.layers.Conv2DTranspose(3,1)(conc5)

softmax_pixelwise = keras.layers.Softmax(axis=3)(cnn9)

G0 = keras.layers.Reshape((272,272,1))(softmax_pixelwise[:,:,:,0])
G1 = keras.layers.Reshape((272,272,1))(softmax_pixelwise[:,:,:,1])
G2 = keras.layers.Reshape((272,272,1))(softmax_pixelwise[:,:,:,2])


hadamard0 = keras.layers.Multiply()([FCN_pr0025_output, G0])
hadamard1 = keras.layers.Multiply()([FCN_pr02_output, G1])
hadamard2 = keras.layers.Multiply()([FCN_pr1_output, G2])

conc6=keras.layers.Concatenate()([hadamard0,hadamard1,hadamard2])


batch10=keras.layers.BatchNormalization(-1)(conc6)
cnn10=keras.layers.Conv2D(64,5,activation=activation)(batch10)
batch11=keras.layers.BatchNormalization(-1)(cnn10)
cnn11=keras.layers.Conv2D(weights[0],3,activation=activation)(batch11)
batch12=keras.layers.BatchNormalization(-1)(cnn11)
cnn12=keras.layers.Conv2D(weights[1],3,activation=activation)(batch12)
batch13=keras.layers.BatchNormalization(-1)(cnn12)
cnn13=keras.layers.Conv2D(weights[2],3,activation=activation)(batch13)
batch14=keras.layers.BatchNormalization(-1)(cnn13)
cnn14=tf.keras.layers.Conv2D(1,1)(batch14)
output=keras.layers.Cropping2D(cropping=3)(cnn14)

model = keras.Model(input_list, outputs=output)
model.summary()
#%%
from DataHandling.features import slices
import xarray as xr 
var=['u_vel','v_vel','w_vel','pr_mix_0025_02_1']
target=['pr0.025_flux_mix_pr0.2_pr1']
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

wandb.init(project="Thesis",notes="Mixture of Experts nonlinear pr0025+02+1, softmaxed")



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


