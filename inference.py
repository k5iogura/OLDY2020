import os
import argparse
import time
import tqdm

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torchvision

from model import MalignancyDetector

def get_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ## DEFINE INPUT AND OUTPUT DATA FOLDERS
    parser.add_argument('--data_path', type=str, default=os.path.join(os.getcwd(), 'input_images'))
    parser.add_argument('--output_folder', type=str, default=os.path.join(os.getcwd(), 'prediction'))

    ## DEFINE MODEL SETTINGS
    parser.add_argument('--backbonename', type=str, default='efficientnet-lite1')
    parser.add_argument('--weights', type=str, default='final.pth')
    parser.add_argument('--image_size', type=int, default=256)

    args = parser.parse_args()

    return args

def open_model(model, modelname):
    """Load model parameters from checkpoint"""
    sd = torch.load(os.path.join(os.getcwd(), 'trained_model', modelname), map_location='cpu')
    model.load_state_dict(sd)

    return model

def run(**kwargs):
    #### RUN CODE ON CUDA IF AVAILABLE
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)

    #### INITIALIZE PRETRAINED EFFICIENTNET
    model = MalignancyDetector(backbone=kwargs['backbonename'], num_classes=2, dropout=0.)
    model = open_model(model, kwargs['weights'])
    model.eval()
    model.to(device)
    #if torch.cuda.is_available():
    #    model.half()

    print('initialized model {}, with {} parameters'.format('MalignancyDetector', sum(p.numel() for p in model.parameters() if p.requires_grad)))

    #### INDEX ALL THE INPUT DATA
    imagelist = os.listdir(kwargs['data_path'])

    #### DEFINE THE AUGMENTATION PARAMETERS
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(kwargs['image_size'], kwargs['image_size'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, inplace=False),
    ])

    #### START INFERENCE
    time_total = 0.
    with torch.no_grad():
        for idx, imagename in tqdm.tqdm(enumerate(imagelist), total=len(imagelist)):
            ### LOAD AND PREPROCESS IMAGE
            path = os.path.join(kwargs['data_path'], imagename)
            img = Image.open(path).convert('RGB')
            Timg = transforms(img).unsqueeze(0).to(device)
            #if torch.cuda.is_available():
            #    Timg = Timg.half() ## Images are set to half precision (float16) to test the quantization performance.

            tstart = time.time()

            ### ANALYSIS OF THE IMAGE BY MODEL
            output = nn.functional.softmax(model(Timg), dim=1)

            tstop = time.time()
            time_total += tstop - tstart

            ### TRANSFORM SEGMENTATION INTO MASK
            ### The second channel of the output is taken, since this channel correlates with malignancies
            prediction = np.array(output[0, 1, :, :].squeeze().unsqueeze(-1).cpu().numpy() * 255, dtype=np.uint8)

            ### CONVERT GRAYSCALE SEGMENTATION TO HEATMAP
            heatmap = cv2.cvtColor(cv2.applyColorMap(prediction, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heatmap = heatmap / 255.

            ### ADD ALPHA CHANNEL TO HEATMAP IN ORDER TO GENERATE A NICER VISUALIZATION
            alphavalue = 0.5
            alpha = np.where(prediction > 128, alphavalue, 0.)
            heatmap_alpha = np.array(np.concatenate((heatmap, alpha), axis=-1) * 255, dtype=np.uint8)
            heatmap_pil = Image.fromarray(heatmap_alpha, mode='RGBA')
            heatmap_pil = heatmap_pil.resize(size=(int(img.size[0]), int(img.size[1])), resample=Image.NEAREST)
            heatmap_pil = heatmap_pil.convert('RGB')

            ### OVERLAY THE HEATMAP OVER THE ORIGINAL INPUT IMAGE
            composite = Image.blend(heatmap_pil, img, 0.8)

            ### SAVE IMAGE TO OUTPUT FOLDER
            composite.save(os.path.join(kwargs['output_folder'], os.path.split(path)[1] + '.jpg')) # images are written as jpg to reduce image size

    print('Total inference time: ', time_total, ' Secondes')

if __name__=='__main__':
    args = get_params()

    print(args.__dict__)
    run(**args.__dict__)
