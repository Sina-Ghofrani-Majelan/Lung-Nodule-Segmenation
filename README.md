# Lung Nodule Segmenation
This repository contains the Keras implementation of my MSc thesis (Iran University Of Science and Technology)
<br/>Thesis tilte:Pulmonary nodule detection Based on Deep Learning
<br/>
<br/>Here is a brief description of my thesis:
<br/>
<br/>
Lung cancer is an aggressive disease resulting in more than one million deaths per year worldwide. Sometimes there is a suspicious tissue in the lungs, referred to as a lung nodule. In the early stages, lung cancer generally manifests in the form of pulmonary nodules, and to determine if someone will develop lung cancer, we have to look for the early stages of malignant pulmonary nodules. Pulmonary nodules have different types and forms, and even all of them are not malignant or cancerous. This significant difference in the shape of nodules, as well as their small size, makes nodule detection more difficult. An example of CT images is shown in Figure 1.
<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/SinaGhofrani1/Lung-Nodule-Segmenation/blob/main/Images/Figure1.jpg" alt> 
    <br/><em>Figure 1- An example of CT images in three different views. From left to right respectively: axial, sagittal, and coronal views. The lung nodule in these images is marked using a red bounding box.</em>
</p>
<br/>
<!--**Align left:**
<p align="left" width="100%">
    <img width="33%" src="https://i.stack.imgur.com/RJj4x.png"> 
</p>-->

<!--**Align center:**
<p align="center" width="100%">
    <img width="33%" src="https://i.stack.imgur.com/RJj4x.png"> 
</p>-->

<!--**Align right:**
<p align="right" width="100%">
    <img width="33%" src="https://i.stack.imgur.com/RJj4x.png"> 
</p>-->
In my thesis, I have tried to study lung cancer screening based on deep learning. Designing a system for lung cancer screening starts with preparing the data and the labels for training the network. A voxel value in a CT image represents the radiodensity of a tissue and is measured with the Hounsfield meter. Therefore, all pixel values of a CT image should be transferred to the Hounsfield meter. By applying a modified normalization filter, morphological erosion and dilation filter, K-means algorithm, and some other techniques, the lung areas are segmented. Therefore, other tissues that do not contain any information about nodules are removed, and the data is ready to be fed to the model. Additionally, the mask labels are obtained based on the nodule coordinates, the minimum diameter of nodules, and the maximum diameter of nodules. Figure 2 shows an example of lung segmentation.
<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/SinaGhofrani1/Lung-Nodule-Segmenation/blob/main/Images/Figure2.jpg" alt> 
    <br/><em>Figure 2- Original CT image and the corresponding segmented lungs (after performing pre-process steps)</em>
</p>
<br/>
In this research, a deep neural network is proposed for lung nodule segmentation using 2D slices of Chest CT images. For the encoder part of this network, the
[I'm an inline-style link](https://www.google.com)
