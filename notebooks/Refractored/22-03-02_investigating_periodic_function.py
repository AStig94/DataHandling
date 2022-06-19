#%%
#from re import X
import tensorflow as tf
import keras

from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils


import numpy as np

x = np.linspace(1,25,25)
x.shape = (-1,5)
x = tf.convert_to_tensor(x)

def wrap_pad(input, size):
    M1 = tf.concat([input[:, -size:], input, input[:, 0:size]], 1)
    M1 = tf.concat([M1[-size:, :], M1, M1[0:size, :]], 0)
    return M1

y = wrap_pad(x,2)

#%%