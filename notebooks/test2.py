#%%
from tensorflow.keras.layers import InputSpec 
from tensorflow.python.keras.utils import conv_utils

padding=5
y=conv_utils.normalize_tuple(padding, 1, 'padding')
input_spec=InputSpec(ndim=3)