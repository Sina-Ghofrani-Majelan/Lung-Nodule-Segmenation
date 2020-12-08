#EDIT HERE##############################

truenoduleweightspath="file:///D:/%23sina/postprocess.pyD:/#sina/truenodule-cnn-weights-improvement.hdf5"
metadatapath="D:/lidc_idri/CSVfiles/LIDC-IDRI_MetaData.csv"
list32path="D:/lidc_idri/CSVfiles/list3.2.csv"
DOIfolderpath='D:/lidc_idri/LIDC-IDRI/'
datafolder='I:/nodule'


########################################

from keras.engine import Layer
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.regularizers import l2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.pyplot as plt
import dicom
import os
import scipy.ndimage
import time
from keras.callbacks import ModelCheckpoint
#import h5py
from sklearn.cluster import KMeans
from skimage import measure, morphology
#from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cell_magic_wand as cmw
#from glob import glob
import random
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from tqdm import tqdm


#Load nodules locations

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

def weighted_log_loss3(y_true, y_pred):
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
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())

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

def gen3(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss and weighted cross entropy
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_true = K.clip(y_true, K.epsilon(), 1 - K.epsilon())
    
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

def gen_dice_loss3(y_true, y_pred):
    return gen3(y_true, y_pred)+weighted_log_loss3(y_true,y_pred)

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



# Load the scans in given folder path
# code sourced from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s, force=True) for s in os.listdir(path) if s.endswith('.dcm')]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]), reverse=True)
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

#convert to ndarray
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def processimage(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    max = np.max(img)
    min = np.min(img)

    middle = img[100:400,100:400] 
    mean = np.mean(middle)  

    #move the underflow bins
    img[img==max]=mean
    img[img==min]=mean
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.0,0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img,np.ones([4,4]))
    dilation = morphology.dilation(eroded,np.ones([10,10]))

    labels = measure.label(dilation)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<475 and B[3]-B[1]<475 and B[0]>40 and B[2]<472:
            good_labels.append(prop.label)
    mask = np.ndarray([512,512],dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation
    return mask*img

def processimagefromfile(ppix):
    processpix=np.ndarray([ppix.shape[0],512,512])
    for i in range(ppix.shape[0]):
#    for i in range(1):
        processpix[i]=processimage(ppix[i])
    return processpix

#predict mask from images
  
def fppredict(images):
#    images=images.reshape(images.shape[0],1,512,512)
    images=np.expand_dims(images,axis=-1)
    num_test=images.shape[0]
    imgs_mask_test = np.ndarray([num_test,400,400,1],dtype=np.float32)
    predictedmask  = np.ndarray([400,400],dtype=np.float32)
    accuracy       = np.ndarray([num_test,1],dtype=np.float32)
    for i in range(num_test):
        fprpredict    = model_fpr.predict([images[i:i+1]],batch_size=1)
        predictedmask = fprpredict[1][0,:,:,1]
#        predictedmask = np.squeeze(model.predict([images[i:i+1]],batch_size=1)[0:1,:,:,0:1])
        mean = np.mean(predictedmask)
        std = np.std(predictedmask)
        predictedmask = predictedmask-mean
        predictedmask = predictedmask/std
        kmeans = KMeans(n_clusters=2).fit(np.reshape(predictedmask,[np.prod(predictedmask.shape),1]))
#        kmeans = KMeans(n_clusters=2).fit(np.reshape(predictedmask[25:375,25:375],[np.prod(predictedmask[25:375,25:375].shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        predictedmask = np.where(predictedmask<threshold,0.0,1.0)
#        predictedmask = morphology.erosion(predictedmask,np.ones([5,5]))
        imgs_mask_test[i]=np.expand_dims(predictedmask,axis=-1)
        accuracy[i]   = fprpredict[0][0][1]
    return imgs_mask_test,accuracy

def senspredictmask(images):
#    images=images.reshape(images.shape[0],1,512,512)
    images=np.expand_dims(images,axis=-1)
    num_test=images.shape[0]
    imgs_mask_test = np.ndarray([num_test,400,400,1],dtype=np.float32)
    predictedmask  = np.ndarray([400,400],dtype=np.float32)
    for i in range(num_test):
        predictedmask = model_sens.predict([images[i:i+1]],batch_size=1)[0,:,:,1]
#        predictedmask = np.squeeze(model.predict([images[i:i+1]],batch_size=1)[0:1,:,:,0:1])
        mean = np.mean(predictedmask)
        std = np.std(predictedmask)
        predictedmask = predictedmask-mean
        predictedmask = predictedmask/std
        kmeans = KMeans(n_clusters=2).fit(np.reshape(predictedmask,[np.prod(predictedmask.shape),1]))
#        kmeans = KMeans(n_clusters=2).fit(np.reshape(predictedmask[25:375,25:375],[np.prod(predictedmask[25:375,25:375].shape),1]))
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        predictedmask = np.where(predictedmask<threshold,0.0,1.0)
#        predictedmask = morphology.erosion(predictedmask,np.ones([5,5]))
        imgs_mask_test[i]=np.expand_dims(predictedmask,axis=-1)
    return imgs_mask_test

#find number of slices where a nodule is detected
def getnoduleindex(imgs_mask_test):
    masksum=[np.sum(np.squeeze(maskslice)) for maskslice in imgs_mask_test]
    return [i for i in range(len(masksum)) if masksum[i]>3]#>5

def nodule_coordinates(nodulelocations,meta):
    slices=nodulelocations["slice no."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    xlocs=nodulelocations["x loc."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    ylocs=nodulelocations["y loc."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][-4:])]]
    nodulecoord=[]
    for i in range(len(slices)):
        nodulecoord.append([slices.values[i],xlocs.values[i],ylocs.values[i]])
    return nodulecoord


#generate nodule or non-nodule labels for mask predictions
def truenodules(noduleindex,masks,nodulecoords):
    label=[]
    for ind in noduleindex:
        for cord in nodulecoords:
            com=scipy.ndimage.center_of_mass(masks[ind])
            if abs(ind-cord[0])<2:
                if abs(com[1]-cord[2])<2 and abs(com[2]-cord[1])<2:
                    label.append(True)
            else:
                label.append(False)
    return label
    
def slicecount(start,end):
    slicecounts=[]
    for i in range(start,end):
        if len(nodule_coordinates(nodulelocations,meta.iloc[i]))>0:
            patient_scan=load_scan(patients[i])
            slicecounts.append(len(patient_scan))
    return slicecounts


meta=pd.read_csv(metadatapath)
meta=meta.drop(meta[meta['Modality']!='CT'].index)
meta=meta.reset_index()

#Get folder names of CT data for each patient
patients=[DOIfolderpath+meta['Patient Id'][i] for i in range(len(meta))]
datfolder=[]
for i in range(0,len(meta)):
    for path in os.listdir(patients[i]):
        if os.path.exists(patients[i]+'/'+path+'/'+meta['Series UID'][i]):
            datfolder.append(patients[i]+'/'+path+'/'+meta['Series UID'][i])
patients=datfolder
nodulelocations=pd.read_csv(list32path)
#noduleimage=np.ndarray([512,512],dtype=np.float32)
#nodulemask=np.ndarray([512,512],dtype=np.bool)
#nodulemaskcircle=np.ndarray([512,512],dtype=np.bool)
thresh=-500
patient_name=[]


model_sens=keras.models.load_model("D:/#sina/m3loss/sinamohammadi/New folder/0.6719_0.3670.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})
model_fpr=keras.models.load_model("D:/#sina/cpm/test1/test2/0.9586_0.8374.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss3,"gen_dice_loss3":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})
#maskedtumor=pix_resampled[int(coord[0])]*cmw_patient
#noduleimages=np.ndarray([10000,512,512],dtype=np.float32)
#noduleimage=np.ndarray([400,400],dtype=np.float32)
#predictedemasks=np.ndarray([len(patients)*8,512,512],dtype=np.float32)
nodulelabels=[]
nodulesensitivity=[]
slicecounts=[]
thresh=-500 #lower HU threshold for nodule segmentation
for i in tqdm(range(len(patients))):
    patient_name = patients[i].split("/")[-3:]
    print("Processing patient#:",i+1,patient_name[-3])
    patient=load_scan(patients[i])
    patient_pix=get_pixels_hu(patient)
    processed_pix = processimagefromfile(patient_pix).astype(np.float32)
    processed_pix = processed_pix[:,57:457,57:457]
    np.save("I:/cpmdatas/preprocessedpix/" + patient_name[-3] + ".npy" , processed_pix)
    senspredicted_mask = senspredictmask(processed_pix)
    np.save("I:/cpmdatas/senspredicts/" + patient_name[-3] + ".npy" , senspredicted_mask)
    noduleindex = getnoduleindex(senspredicted_mask)
    noduleindex = np.array(noduleindex)
    np.save("I:/cpmdatas/sensindexes/" + patient_name[-3] + ".npy" , noduleindex)
    fp_predict_seg,accuracy_fp=fppredict(processed_pix)
    np.save("I:/cpmdatas/fppredictseg/" + patient_name[-3] + ".npy" , fp_predict_seg)
    np.save("I:/cpmdatas/accuracyfp/" + patient_name[-3] + ".npy" , accuracy_fp)
    
        
        
        
        
        
        
        
        
        
        
        
        
'''        
        
    patient_name = patients[i].split("/")[-3:]
    print("Processing patient#:",i,patient_name[-3])
    coord=nodule_coordinates(nodulelocations,meta.iloc[i])
    if len(coord)>0:
        patient=load_scan(patients[i])
        patient_pix=get_pixels_hu(patient)
#        slicecounts.append(len(patient))
        processed_pix = processimagefromfile(patient_pix)
        predictedmask = predictmask(processed_pix[:,57:457,57:457])
        noduleindex = getnoduleindex(predictedmask)        
        
        
        
        
        
       
        radius=nodulelocations["eq. diam."][nodulelocations.index[nodulelocations["case"]==int(meta["Patient Id"][i][-4:])]]
        #################    
        labels = np.zeros(len(noduleindex)).astype(np.bool)
        cordlabels=np.zeros(len(coord)).astype(np.bool)
        #################
        for j,cord in enumerate(coord): #loop through labeled nodules
            if radius.iloc[j]>0:
                for k,ind in enumerate(noduleindex): #loop through detected nodules
                    if abs(ind-cord[0])<2:
                        nodulemask = cmw.cell_magic_wand(-patient_pix[int(ind)],[int(cord[2]),int(cord[1])],2,int(radius.iloc[j])+2)
                        nodulepix=nodulemask[57:457,57:457]
                        nodulemask=nodulepix.astype(np.bool)
                        if np.sum(nodulemask*np.squeeze(predictedmask[ind]))>1:
                            print("Nodule Detected at slice#",ind,"with actual coord",cord)
                        
                nodulemask = cmw.cell_magic_wand(-patient_pix[int(cord[0])],[int(cord[2]),int(cord[1])],2,int(radius.iloc[j])+2)
#                nodulepix=nodulemask[57:457,57:457]*patient_pix[cord[0],57:457,57:457]
#                nodulepix[nodulepix<thresh]=0
#                nodulepix[nodulepix!=0]=1
#                nodulemask=nodulepix.astype(np.bool)
                nodulepix=nodulemask[57:457,57:457]
                nodulemask=nodulepix.astype(np.bool)
                for k,ind in enumerate(noduleindex): #loop through detected nodules
                    if abs(ind-cord[0])<2:
                        if np.sum(nodulemask*np.squeeze(predictedmask[ind]))>1:
                            print("Nodule Detected at slice#",ind,"with actual coord",cord)
                            labels[k] = True
                            cordlabels[j] = True
#
#        for j in range(len(coord)):
#            nodulesensitivity.append(cordlabels[j])
            
        for k in range(len(noduleindex)):
            nodulelabels.append(labels[k])
            noduleimage=processed_pix[noduleindex[k],57:457,57:457]
            #np.save('D:/fp_reduction/noduleimage_' + patient_name[-3] + '_slice_' + str(noduleindex[k]) + '_' + str(noduleindex[k]) + str(labels[k]) + '.npy',noduleimage)
            
        coordtruecount+=len(cordlabels[cordlabels==True])
        print('number of all noduls:' + str(len(coord)))
        print('number of detected noduls:' + str(len(cordlabels[cordlabels==True])))
        noduletruecount+=len(labels[labels==True])
        coordcount+=len(coord)
        nodulecount+=len(noduleindex)
        print("Sensitivity:",coordtruecount/coordcount)
        print("Specificity:",noduletruecount/nodulecount)
#    elapsed_time=time.time()-start_time
#    totaltime=elapsed_time/(i+1)*200
#print(elapsed_time)
#noduleimages=noduleimages[:len(nodulelabels)]



#noduleimages=noduleimages[:len(nodulelabels)]
#noduleimages=noduleimages.reshape([noduleimages.shape[0],1,512,512])
#nodulelabels=np.array(nodulelabels)
#np.save("noduleimages197-399.npy",noduleimages)
#np.save("nodulelabels197-399.npy",nodulelabels)


#noduleimages=np.load("noduleimages0-192.npy")
#nodulelabels=np.load("nodulelabels0-192.npy")
#noduleimages=np.concatenate((noduleimages,np.load("noduleimages197-399.npy")))
#nodulelabels=np.concatenate((nodulelabels,np.load("nodulelabels197-399.npy")))



TP=len([nl for nl in nodulelabels if nl==True])
FP=len([nl for nl in nodulelabels if nl==False])
print("Number of True Positive nodules:",TP)
print("Number of False Positive nodules:",FP)
print("# of FPs per TP",FP/TP)

noduleimagestrue=np.load("noduleimagesv3.npy")
falseind=[i for i in range(len(nodulelabels)) if nodulelabels[i]==False]
random.shuffle(falseind)
falseind=falseind[:noduleimagestrue.shape[0]]
noduleimagesfalse=np.array([noduleimages[i] for i in falseind])
del noduleimages
noduleimagesbalanced=np.concatenate((noduleimagestrue,noduleimagesfalse.reshape(noduleimagesfalse.shape[0],1,512,512)),axis=0)
nodulelabelsbalanced=[True]*noduleimagestrue.shape[0]+[False]*noduleimagesfalse.shape[0]
nodulelabelsbalanced=to_categorical(nodulelabelsbalanced,2)
del noduleimagestrue,noduleimagesfalse
Xtrain, Xtest, Ytrain, Ytest = train_test_split(noduleimagesbalanced,nodulelabelsbalanced,test_size=.30)
del noduleimagesbalanced, nodulelabelsbalanced



#classify as nodule or non-nodule
input_shape=(1,512,512)
num_classes=2
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=Adam(lr=1e-5),
              metrics=['accuracy'])


#model.load_weights(truenoduleweightspath)

checkpoint = ModelCheckpoint(truenoduleweightspath, monitor='loss', verbose=1, save_best_only=True)

history=model.fit(Xtrain, Ytrain, batch_size=4, epochs=20, verbose=1, shuffle=True, callbacks=[checkpoint], validation_data=(Xtest,Ytest))

XtestTrue=np.array([Xtest[i] for i in range(Ytest.shape[0]) if Ytest[i,1]==1])
YtestTrue=np.array([Ytest[i] for i in range(Ytest.shape[0]) if Ytest[i,1]==1])
XtestFalse=np.array([Xtest[i] for i in range(Ytest.shape[0]) if Ytest[i,1]==0])
YtestFalse=np.array([Ytest[i] for i in range(Ytest.shape[0]) if Ytest[i,1]==0])

scoretrue=model.evaluate(XtestTrue,YtestTrue)
scorefalse=model.evaluate(XtestFalse,YtestFalse)


print("Sensitivity:", 0.7477) #Evaluated from 2TrainUnet.ipynb
print("FP Rate/slice:", len(falseind)/(sum(slicecounts+slicecounts2)))
print("FP Rate/slice after nodule classification:", len(falseind)*(1-scorefalse[1])/(sum(slicecounts+slicecounts2)))
print("Sensitivity after nodule classification:", 0.7477*scoretrue[1])

'''