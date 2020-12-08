# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:59:47 2018

@author: DeepPC_IUST
"""
import keras
from data_generator_sequence_table import Data_Generator
import tables
import numpy as np
#from Unet_Model import Unet_model
#from input400 import sinamodel
hdf5_file = tables.open_file("D:/#sina/fullwithsmallnodule__data_3slice_400.hdf5", mode='r+')
#import mobilev1
#from Model_Sina2_Final import Model_Bahri
train_idx = np.load("D:/#sina/fullwithsmallnodule_train_idx_3slice_400.npy")
val_idx   = np.load("D:/#sina/fullwithsmallnodule_val_idx_3slice_400.npy")

datagen_train = Data_Generator(hdf5_file, train_idx, batch_size=12,horizontal_flip=True, vertical_flip=True, rotation_range=6, zoom_range=0, seed=0)
datagen_val   = Data_Generator(hdf5_file, val_idx, batch_size=12, seed=0 , val=True)

###############################################################################
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D
from keras.layers import Input, concatenate, UpSampling2D, BatchNormalization,AveragePooling2D,SeparableConv2D
from keras.optimizers import Adam
#from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.regularizers import l2
#import h5py
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc
from keras.engine import Layer
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.regularizers import l2

K.set_image_data_format('channels_last')
#Code sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
smooth = 1.0
width = 32
weight_decay=5e-5
def dice_coef(y_true, y_pred):
    
   
    
    y_pred = y_pred[...,1]
    y_true = y_true[...,1]

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
def weighted_log_loss2(y_true, y_pred):
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # weights are assigned in this order : normal,necrotic,edema,enhancing 
    weights=np.array([1,7])
    weights = K.variable(weights)
    loss = y_true * K.log(y_pred) * weights
    loss = K.mean(-K.sum(loss, -1))
    return loss

def gen2(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_true_f = K.reshape(y_true,shape=(-1,2))
    y_pred_f = K.reshape(y_pred,shape=(-1,2))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights

    return GDL

 
def gen_dice_loss2(y_true, y_pred):
    return gen2(y_true, y_pred)+weighted_log_loss2(y_true,y_pred)

#def dice_coef_loss(y_true, y_pred):
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
##    spec = (intersection - K.sum(y_true_f)) / (K.sum(y_pred_f) - K.sum(y_true_f))
##    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#    acc  = (2. * intersection - K.sum(y_true_f)) / (K.sum(y_pred_f))
##    sensitivity = intersection / (2. * intersection - K.sum(y_true_f))
#    return (1. - (acc))


#model=Model_Bahri([400,400,1], 2)
model= keras.models.load_model("D:/#sina/m3loss/sinamohammadi/m8438.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})    
model.compile(optimizer=keras.optimizers.SGD(lr=1e-7,momentum=0.9), loss=gen_dice_loss2, metrics= [dice_coef])
filepath = "D:/#sina/m3loss/sinamohammadi/"
checkpoint = ModelCheckpoint(filepath+"{val_dice_coef:.4f}_{val_loss:.4f}.hdf5", monitor= "val_dice_coef", mode = "max", save_best_only= True,  verbose=1)
logger     = CSVLogger(filepath+'01_sequence.log',append=True)
#model.summary()
history = model.fit_generator(datagen_train, epochs=200, callbacks=[checkpoint,logger], validation_data=datagen_val)


