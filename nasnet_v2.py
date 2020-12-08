#@title
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 01:23:04 2019

@author: Sina
"""

import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import  Conv2D , BatchNormalization , Activation
from keras import backend as K
from keras.layers import Dense,Reshape,Permute,multiply ,GlobalAveragePooling2D
from keras import callbacks
K.set_image_data_format("channels_last")
import cv2
from glob import glob     
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Concatenate
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras.utils import conv_utils
from keras import layers
import keras
from keras import backend as K

class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def GCN(x,num_f,k):
  x1=keras.layers.Conv2D(num_f,(k,1),strides=(1,1),padding='same')(x)
  x1=keras.layers.Conv2D(num_f,(1,k),strides=(1,1),padding='same')(x1)
  x2=keras.layers.Conv2D(num_f,(1,k),strides=(1,1),padding='same')(x)
  x2=keras.layers.Conv2D(num_f,(k,1),strides=(1,1),padding='same')(x2)
  xs=keras.layers.add([x1,x2])
  return xs

def comb(x,num_f):
    x=squeeze_excite_block(x, ratio=8)

    
    b0 = Conv2D(num_f, (1, 1), padding='same')(x)    
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    
    
    b1 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=3)(x)
    b1 = BatchNormalization(epsilon=1e-5)(b1)
    b1 = Activation('relu')(b1)
    
    b2 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=6)(x)
    b2 = BatchNormalization(epsilon=1e-5)(b2)
    b2 = Activation('relu')(b2)
    

    b3=GCN(x,num_f,19)
    b3 = BatchNormalization(epsilon=1e-5)(b3)
    b3 = Activation('relu')(b3)
    x = Concatenate()([b0, b1, b2, b3])

    x=squeeze_excite_block(x, ratio=8)
    x = Conv2D(num_f, (1, 1), padding='same')(x)    
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x
  

  
def comb_last(x,num_f):
    x=squeeze_excite_block(x, ratio=8)

    
    b0 = Conv2D(num_f, (1, 1), padding='same')(x)    
    b0 = BatchNormalization(epsilon=1e-5)(b0)
    b0 = Activation('relu')(b0)
    
    b1 = Conv2D(num_f, (3, 3), padding='same')(x)    
    b1 = BatchNormalization(epsilon=1e-5)(b1)
    b1 = Activation('relu')(b1)
    
    b2 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=2)(x)
    b2 = BatchNormalization(epsilon=1e-5)(b2)
    b2 = Activation('relu')(b2)
    
    b3 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=3)(x)
    b3 = BatchNormalization(epsilon=1e-5)(b3)
    b3 = Activation('relu')(b3)
    
    b4 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=4)(x)
    b4 = BatchNormalization(epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)

    b5 = keras.layers.SeparableConv2D(num_f, (3,3), strides=(1, 1), padding='same', dilation_rate=5)(x)
    b5 = BatchNormalization(epsilon=1e-5)(b5)
    b5 = Activation('relu')(b5)    

    b6=GCN(x,num_f,13)
    b6 = BatchNormalization(epsilon=1e-5)(b6)
    b6 = Activation('relu')(b6)
    
    x = Concatenate()([b0, b1, b2, b3, b4,b5,b6])
    x=squeeze_excite_block(x, ratio=8)
    x = Conv2D(num_f, (1, 1), padding='same')(x)    
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Activation('relu')(x)
    return x 


y=keras.applications.nasnet.NASNetMobile(input_shape=(400,400,1), include_top=False, weights=None)
x=comb_last(y.output,192)
x=BilinearUpsampling(output_size=(25,25))(x)
xc=Concatenate()([y.layers[537].output,y.layers[549].output])
xc =comb(xc,192)
xs=keras.layers.Concatenate()([x,xc])
xs=squeeze_excite_block(xs, ratio=8)
x = keras.layers.Conv2D(128, (3,3), strides=(1, 1), padding='same')(xs)
x = BatchNormalization(epsilon=1e-5)(x)
x = Activation('relu')(x)
x=BilinearUpsampling(output_size=(50,50))(x)
xc=Concatenate()([y.layers[300].output,y.layers[301].output])
xc =comb(xc,128)
xs=keras.layers.Concatenate()([x,xc])
xs=squeeze_excite_block(xs, ratio=8)
x = keras.layers.Conv2D(96, (3,3), strides=(1, 1), padding='same')(xs)
x = BatchNormalization(epsilon=1e-5)(x)
x = Activation('relu')(x)
x=BilinearUpsampling(output_size=(100,100))(x)
xc=Concatenate()([y.layers[63].output,y.layers[64].output])
xc =comb(xc,96)
xs=keras.layers.Concatenate()([x,xc])
xs=squeeze_excite_block(xs, ratio=8)
x = keras.layers.Conv2D(64, (3,3), strides=(1, 1), padding='same')(xs)
x = BatchNormalization(epsilon=1e-5)(x)
x = Activation('relu')(x)
x=BilinearUpsampling(output_size=(199,199))(x)
x = Conv2D(64, (3, 3), padding='same')(x)    
x = BatchNormalization(epsilon=1e-5)(x)
x = Activation('relu')(x)
x=BilinearUpsampling(output_size=(400,400))(x)
x = Conv2D(2, (1, 1), padding='same')(x)
x=Activation('softmax')(x)
model=keras.models.Model(y.input,x)