### Deliverable TUe-Olympus Tokyo 15-12-2020 -- Barrett's Neoplasia, Efficientnet-lite + RASSP-lite Segmentation Model

The contents of these deliverables only contain the inference code for the translation to FPGA. If Olympus wishes to
receive the training code as well, we will provide that in a later stage.

## Contents
# Folders
'trained_model' : This folder contains the trained weights of the Efficientnet-lite + RASSP-lite model.
'input_images': This is an empty folder in which the user can add images for running inference of the model
'prediction': This is an empty folder in which 'inference.py' scripts writes the output segmentation masks of the model

# Files
'model.py': This is the main file to load the Efficientnet-lite + RASSP-lite segmentation model.
If the user runs 'python model.py', the user can run an inference speed test, to measure the FPS.
'backbone.py': This scripts contains the main building blocks for the EfficientNet-lite encoder.
'backbone_utils.py': This script contains tools for the EfficientNet-lite for the efficient scaling of the architectures.
'inference.py': This script contains the inference module for analyzing endoscopic images. N.B. the user needs to add images to the 'input_images folder'.
To run this scripts, the user only needs to run 'python inference.py', necessary settings are defined within the script.
Data augmentation will be performed automatically, this entails resizing of the image to 256x256 and normalization.
The model only returns a segmentation mask for simplicity, while for training the model are trained with a classification + segmentation head as output.
The results of the Meeting on 11-december-2020 in slide 26 were generated with this script.

## Dependencies
Python 3.6.9
Pytorch 1.6.0






