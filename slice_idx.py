# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:38:22 2018file:///D:/%23sina/model.py

@author: DeepPC_IUST
"""

import os
import tables
import numpy as np
from glob import glob

hdf5_file = tables.open_file('fullwithsmallnodule__data_3slice_400.hdf5', mode='r+')
data_storage        = hdf5_file.root.data

all_indexes = np.arange(len(data_storage))

np.random.seed(85)
np.random.shuffle(all_indexes)

split_rate = 0.8
split_point = int(np.ceil(len(data_storage) * split_rate))

train_idx = all_indexes[:split_point]
val_idx   = all_indexes[split_point:]

np.save("fullwithsmallnodule_train_idx_3slice_400.npy", train_idx)
np.save("fullwithsmallnodule_val_idx_3slice_400.npy", val_idx)