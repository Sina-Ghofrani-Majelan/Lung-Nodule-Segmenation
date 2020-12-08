# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:11:16 2019

@author: DeepPC_IUST
"""
import numpy as np
from skimage import  morphology
from sklearn.cluster import KMeans
import keras
from keras import backend as K
smooth = 1.0
width = 32
#import Model_Sina2_V3
#from Model_Sina2_V3 import Model_Bahri
from keras.engine import Layer
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tables
from glob import glob

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



val_idx   = np.load("D:/#sina/fullwithsmallnodule_val_idx_3slice_400.npy")
hdf5_file = tables.open_file('D:/#sina/fullwithsmallnodule__data_3slice_400.hdf5', mode='r+')
data_storage        = hdf5_file.root.data
truth_storage       = hdf5_file.root.truth
subject_ids_storage = hdf5_file.root.subject_ids



inputdata = np.expand_dims(data_storage[val_idx[5]],axis=-1)
inputdata = np.expand_dims(inputdata,axis=0)
ex=data_storage[val_idx[5]]
actualmask = truth_storage[val_idx[5]]

#inputdata = np.expand_dims(data_storage[1],axis=-1)
#inputdata = np.expand_dims(inputdata,axis=0)
#ex=data_storage[1]
#actualmask = truth_storage[1]

hdf5_file.close()    



##inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0041_1.3.6.1.4.1.14519.5.2.1.6279.6001.212292046142156223429795319169_1.3.6.1.4.1.14519.5.2.1.6279.6001.138813197521718693188313387015_24_0.npy')
##inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0037_1.3.6.1.4.1.14519.5.2.1.6279.6001.250770014904528873190814943829_1.3.6.1.4.1.14519.5.2.1.6279.6001.219909753224298157409438012179_33_0.npy')
#inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0029_1.3.6.1.4.1.14519.5.2.1.6279.6001.788972240715000723677133060452_1.3.6.1.4.1.14519.5.2.1.6279.6001.264090899378396711987322794314_93_0.npy')
##inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0001_1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192_42_0.npy')
##inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0005_1.3.6.1.4.1.14519.5.2.1.6279.6001.190188259083742759886805142125_1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225_44_0.npy')
##inputdata=np.load('D:/processed/image/noduleimage_LIDC-IDRI-0002_1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329_1.3.6.1.4.1.14519.5.2.1.6279.6001.619372068417051974713149104919_76_0.npy')
#inputdata=np.expand_dims(inputdata[57:457,57:457],axis=-1)
#inputdata=np.expand_dims(inputdata,axis=0)
#ex=np.squeeze(inputdata[0])
##model=keras.models.load_model("D:/#sina/-0.7600_-0.7065665.hdf5", custom_objects={"dice_coef_loss":dice_coef_loss,"dice_coef" :dice_coef})
##model=keras.models.load_model("D:/#sina/0.2442_0.34920043_0.7012.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss2,"dice_coef" :dice_coef})
model=keras.models.load_model("D:/#sina/m3loss/sinamohammadi/New folder/0.6719_0.3670.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})
predictmask=model.predict(inputdata,batch_size=1)
predictmask=predictmask[0,:,:,1]
#predictmask=predictmask[0,:,:,0]
#
##actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0041_1.3.6.1.4.1.14519.5.2.1.6279.6001.212292046142156223429795319169_1.3.6.1.4.1.14519.5.2.1.6279.6001.138813197521718693188313387015_24_0.npy')
##actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0037_1.3.6.1.4.1.14519.5.2.1.6279.6001.250770014904528873190814943829_1.3.6.1.4.1.14519.5.2.1.6279.6001.219909753224298157409438012179_33_0.npy')
#actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0029_1.3.6.1.4.1.14519.5.2.1.6279.6001.788972240715000723677133060452_1.3.6.1.4.1.14519.5.2.1.6279.6001.264090899378396711987322794314_93_0.npy')
##actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0001_1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178_1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192_42_0.npy')
##actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0005_1.3.6.1.4.1.14519.5.2.1.6279.6001.190188259083742759886805142125_1.3.6.1.4.1.14519.5.2.1.6279.6001.129007566048223160327836686225_44_0.npy')
##actualmask=np.load('D:/processed/label/nodulemask_LIDC-IDRI-0002_1.3.6.1.4.1.14519.5.2.1.6279.6001.490157381160200744295382098329_1.3.6.1.4.1.14519.5.2.1.6279.6001.619372068417051974713149104919_76_0.npy')
#actualmask=actualmask[57:457,57:457]
##predictmask=np.load('C:/Users/DeepPC_IUST/Desktop/img.npy')
#
#
img = predictmask
mean = np.mean(img)
std = np.std(img)
img = img-mean
img = img/std
#max = np.max(img)
#min = np.min(img)
#img[img==max]=mean
#img[img==min]=mean
kmeans = KMeans(n_clusters=2).fit(np.reshape(img,[np.prod(img.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,0.0,1.0)
eroded = morphology.erosion(thresh_img,np.ones([5,5]))
#dilation = morphology.dilation(thresh_img,np.ones([10,10]))
#dilation = morphology.dilation(eroded,np.ones([10,10]))

#
#
######################CPM TEST###########################


model_fpr=keras.models.load_model("D:/#sina/cpm/test1/test2/0.9586_0.8374.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss3,"gen_dice_loss3":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})
FPR_res=model_fpr.predict(inputdata,batch_size=1)
fpr_ex=FPR_res[1][0,...,1]















################################################

###############################################################
############################################################
###############################jsrt###########################
###############################################################
############################################################





#input_filename = "I:/Lung_DataSet/jsrt/Nodule154images/JPCLN039.IMG"
#shape = (2048, 2048) # matrix size
#dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)
#output_filename = "I:/Lung_DataSet/jsrt/Nodule154images/JPCLN039.PNG"
#
## Reading.
#fid = open(input_filename, 'rb')
#data = np.fromfile(fid, dtype)
#image = data.reshape(shape)
#
## Display.
#plt.imshow(image, cmap = "gray")
#plt.savefig(output_filename)
#plt.show()
#

#########################################################################
##############################LUNA####################################
#########################################################################
import cell_magic_wand as cmw
import pandas
import glob
import ntpath
import SimpleITK as sitk
import numpy as np
from skimage import measure, morphology

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)

    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))### z,y,x  Origin in world coordinates (mm)///The image origin is associated with the coordinates of the first pixel in the image.///left and down pixel
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))# spacing of voxels in world coor. (mm)////Pixel spacing is measured between the pixel centers and can be different along each dimension
################################################################### z,y,x  Origin in world coordinates (mm)///the voxel lattice نسبت به physical space orientation///جوابش یک ماتریس است
    return numpyImage, numpyOrigin, numpySpacing

originalSubsetDirectoryBase =  r'I:/LUNA16_Challange-master/source/subset'
annotations = r'I:/LUNA16_Challange-master/source/CSVFILES/annotations.csv'
folderName = originalSubsetDirectoryBase + str(0)
allpatientIDs = []
patientnames  = []
def worldToVoxelCoord(worldCoord, origin, spacing):
    strechedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = strechedVoxelCoord / spacing
    return voxelCoord
allpatientIDs = glob.glob(folderName + "/"+ "*.mhd")
numpyImage, numpyOrigin, numpySpacing = load_itk_image(allpatientIDs[37])
for src_path in allpatientIDs:
    patientname = ntpath.basename(src_path).replace(".mhd", "")
    patientnames.append(patientname)
annotationsList = pandas.read_csv(annotations)
voxelWorldCord = np.asarray([float(annotationsList["coordZ"][630]), float(annotationsList["coordY"][630]), float(annotationsList["coordX"][630])])
newGeneratedCord = worldToVoxelCoord(voxelWorldCord, numpyOrigin,numpySpacing)

ex1 = numpyImage[int(newGeneratedCord[0])]

def processimage(img):
    #function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    #Standardize the pixel values
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    #plt.hist(img.flatten(),bins=200)
    #plt.show()
    #print(thresh_img[366][280:450])
    middle = img[100:400,100:400] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
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



def processimagefromfile(numpyImage):
    processpix=np.ndarray([numpyImage.shape[0],512,512])
    for i in range(numpyImage.shape[0]):
#    for i in range(1):
        processpix[i]=processimage(numpyImage[i])
    return processpix


processed_pix = processimagefromfile(numpyImage)


ex = processed_pix[int(newGeneratedCord[0]),57:457,57:457]
inputdata=np.expand_dims(ex,axis=-1)
inputdata=np.expand_dims(inputdata,axis=0)

model=keras.models.load_model("D:/#sina/m3loss/FINAL/0.8419_0.7125.hdf5", custom_objects={"gen_dice_loss2":gen_dice_loss2,"dice_coef" :dice_coef, "BilinearUpsampling":BilinearUpsampling})
predictmask=model.predict(inputdata,batch_size=1)
predictmask=predictmask[0,:,:,1]
#predictmask=predictmask[0,:,:,0]


img = predictmask
mean = np.mean(img)
std = np.std(img)
img = img-mean
img = img/std
#max = np.max(img)
#min = np.min(img)
#img[img==max]=mean
#img[img==min]=mean
kmeans = KMeans(n_clusters=2).fit(np.reshape(img,[np.prod(img.shape),1]))
centers = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centers)
thresh_img = np.where(img<threshold,0.0,1.0)



#
#
#nodulemask = cmw.cell_magic_wand(-patient_pix[int(cord[0])],[int(cord[2]),int(cord[1])],2,int(radius.iloc[j])+2)
#






