# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 21:41:12 2018

@author: DeepPC_IUST
"""

import os
import tables
import numpy as np
from glob import glob

images = glob("I:/nodule/image/*.npy")
labels = glob("I:/nodule/label/*.npy")


#for i, j in zip(images , labels):
#    
#    if (os.path.basename(i)[12:]!=os.path.basename(j)[11:]):
#        print("not same")
#    
image_shape = (400,400)

hdf5_file = tables.open_file('fullwithsmallnodule__data_3slice_400.hdf5', mode='w')

filters = tables.Filters(complevel=5, complib='blosc')
data_shape  = tuple([0] + list(image_shape))
truth_shape = tuple([0] + list(image_shape))

data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape,
                                           filters=filters, expectedrows=len(images))
truth_storage = hdf5_file.create_earray(hdf5_file.root, 'truth', tables.UInt8Atom(), shape=truth_shape,
                                            filters=filters, expectedrows=len(labels))

for i, j in zip(images , labels):
    image = np.load(i)[57:457,57:457]
    label = np.load(j)[57:457,57:457]
    
    data_storage.append(image[np.newaxis,...])
    truth_storage.append(label[np.newaxis,...])
    
    
    
subject_ids = [os.path.basename(i)[12:] for i in images]
hdf5_file.create_array(hdf5_file.root, 'subject_ids', obj=subject_ids)
hdf5_file.close()    
