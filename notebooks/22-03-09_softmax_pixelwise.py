#%%
#from re import X
import tensorflow as tf
import keras

from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils


import numpy as np
x = np.linspace(1,9,9)
x.shape = (-1,3)
x = tf.convert_to_tensor(x,dtype='double')


y = np.linspace(10,18,9)
y.shape = (-1,3)
y = tf.convert_to_tensor(y,dtype='double')

stack=tf.stack([x,y])

tf.nn.softmax(stack,axis=0)



# %%
