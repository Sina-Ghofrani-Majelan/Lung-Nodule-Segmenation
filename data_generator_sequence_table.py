# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 19:33:55 2018

@author: M3hrdad
"""

import numpy as np
from keras.utils import Sequence , to_categorical
try:
    import scipy
    # scipy.linalg cannot be accessed until explicitly imported
    from scipy import linalg
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0.):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.

    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [scipy.ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=1,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class Data_Generator(Sequence):
    
    def __init__(self, hdf5_file,
                 slice_idx,
                 batch_size=16,
                 n_channels=1,
                 shuffle=True, 
                 horizontal_flip=False,
                 vertical_flip=False,
                 rotation_range=0,
                 zoom_range=0.0,
                 seed=0,
                 val=False):
    
    
        self.hdf5_file       = hdf5_file 
        self.data_storage    = self.hdf5_file.root.data
        self.truth_storage   = self.hdf5_file.root.truth
        self.batch_size      = batch_size
        self.slice_idx       = slice_idx
        self.n_channels      = n_channels
        self.shuffle         = shuffle
        self.val             = val
        self.seed            = seed
        self.horizontal_flip = horizontal_flip
        self.vertical_flip   = vertical_flip
        self.rotation_range  = rotation_range
        
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))
            
        self.on_epoch_end()

    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
    
    
    def apply_transform(self, x, transform_parameters):
        
        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                           transform_parameters.get('tx', 0),
                           transform_parameters.get('ty', 0),
                           transform_parameters.get('shear', 0),
                           transform_parameters.get('zx', 1),
                           transform_parameters.get('zy', 1),
                           row_axis=0,
                           col_axis=1,
                           channel_axis=2)
        if transform_parameters.get('flip_horizontal', False):
            x = self.flip_axis(x, 1)
        if transform_parameters.get('flip_vertical', False):
            x = self.flip_axis(x, 0)            
        return x
        
    def get_random_transform(self):
    
        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range,self.rotation_range)    
        else:
            theta = 0            
 
        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0],self.zoom_range[1], 2)
            
        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip    
        flip_vertical   = (np.random.random() < 0.5) * self.vertical_flip
        
        transform_parameters = {'flip_horizontal': flip_horizontal, 'flip_vertical':flip_vertical , 'theta': theta, 'zx': zx, 'zy': zy}
    
        return transform_parameters

    def __len__(self):
#        if self.target_dir == None:
#            lenght = int(np.floor( (len(self.slice_idx)*self.shape[1]) / self.batch_size))
#        else:
#            lenght = int(np.floor( len(self.target) / self.batch_size))
#        return lenght
        return int(np.floor( len(self.indexes) / self.batch_size))
    
    
    def __getitem__(self, index):

        # Generate indexes of the batch
        idx = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X_batch, Y_batch = self.data_load_and_preprocess(idx)

        return X_batch, Y_batch

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.slice_idx 
        
        if self.shuffle == True:
            np.random.seed(self.seed)
            np.random.shuffle(self.indexes)
            if self.val == False:
                self.seed = self.seed + 10
            
            
    def data_load_and_preprocess(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        Slice_Batch = []
        Label_batch = []

        # Generate data
        for i in idx:
            Slice_ID = i
            Slice    = np.expand_dims(self.data_storage[Slice_ID],axis=-1)
            Label    = np.expand_dims(self.truth_storage[Slice_ID],axis=-1)
            Slice_and_Label = np.concatenate((Slice,Label) , axis=-1)
            #Slice_and_Label = np.load(self.brains_dir[Brain_ID])[:,Slice_ID,:,:]############
            #Slice_and_Label = self.normalize_slice(Slice_and_Label)
            params          = self.get_random_transform()
            #Slice_and_Label = Slice_and_Label.transpose(1,2,0)
            Slice_and_Label = self.apply_transform(Slice_and_Label, params)
            Slice           = Slice_and_Label[...,0:1]
            Label           = to_categorical(Slice_and_Label[...,1],2)
            
            Slice_Batch.append(Slice)
            Label_batch.append(Label)
            
        return np.array(Slice_Batch), np.array(Label_batch)