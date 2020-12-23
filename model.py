import os,sys

import torch.nn as nn
import torch
import torch.nn.functional as F
from pdb import set_trace
from inspect import getmembers

from backbone import EfficientNet as EffNet
from dyna import *

def open_model(model, modelname):
    """Open model parameters from checkpoint"""
    sd = torch.load(os.path.join(os.getcwd(), 'trained_model', modelname) , map_location='cpu')
    model.load_state_dict(sd)

    return model

class R_ASPP_module(nn.Module):
    """Reduced Atrous Spatial Pyramid Pooling Lite Segmentation Head.
    Args:
        in_channels_x: Number of channels originating from the feature extractor
        in_channels_f: Number of channels originating from the skip channel
        num_classes: number of output channels. PLEASE NOTE, the OUTPUT is a concatenation of featuremaps and should be divisible by 2.
    References:
        [1] https://arxiv.org/pdf/1802.02611.pdf  (Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation)
        [2] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """
    def __init__(self, in_channels_x, in_channels_f, num_classes):
        super(R_ASPP_module, self).__init__()

        assert not num_classes % 2

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels_x, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
        )

        self._act = nn.ReLU6()
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.layer2 = nn.Conv2d(in_channels_x, 128, kernel_size=1, stride=1)

        self.out_conv1 = nn.Conv2d(128, num_classes//2, kernel_size=1, stride=1)
        self.out_conv2 = nn.Conv2d(in_channels_f, num_classes//2, kernel_size=1, stride=1)

        self.hsigmoid = nn.Hardsigmoid()

        self._init_weight()

    def forward(self, x, feature):

        print("<<< START R_ASPP >>>")
        print("IN-1 {} IN-2 {}".format(x.shape, feature.shape))
        print("<<< START ROUTE-1-1 >>>")
        print("IN {}".format(x.shape))
        x_temp1 = self._act(self.layer1(x))
        anlz_block(self.layer1,1)
        anlz_submod(self._act)
        print("GO {}".format(x_temp1.shape))
        print("<<< ENDED ROUTE-1-1 >>>\n")

        print("<<< START ROUTE-1-2 >>>")
        print("IN {}".format(x.shape))
        # Squeeze and excitation module for Segmentation head
        x_temp2 = self.avgpool(x)
        x_temp2 = self.layer2(x_temp2)

        x_temp2_weight = self.hsigmoid(x_temp2) # sigmoid function is replaced with the Hardsigmoid function.
        x_temp2_weight = F.interpolate(x_temp2_weight, x_temp1.size()[2:], mode='bilinear', align_corners=False)
        [anlz_submod(s) for s in (self.avgpool, self.layer2, self.hsigmoid, F.interpolate)]
        print("GO {}".format(x_temp2_weight.shape))
        print("<<< ENDED ROUTE-1-2 >>>\n")

        print("<<< START ROUTE-1-AGRIGATE >>>")
        print("IN-1 {} IN-2 {}".format(x_temp2_weight.shape, x_temp1.shape))
        out = x_temp2_weight * x_temp1
        print("PRODUCT-GO {}".format(out.shape))
        out = F.interpolate(out, feature.size()[2:], mode='bilinear', align_corners=False)

        # Compress feature maps to number of classes
        out = self.out_conv1(out)
        #feature = self.out_conv2(feature)
        [anlz_submod(s) for s in (F.interpolate, self.out_conv1)]
        print("GO {}".format(out.shape))
        print("<<< ENDED ROUTE-1-AGRIGATE >>>\n")

        print("<<< START ROUTE-2 >>>")
        print("IN {}".format(feature.shape))
        feature = self.out_conv2(feature)
        anlz_submod(self.out_conv2)
        print("GO {}".format(feature.shape))
        print("<<< ENDED ROUTE-2 >>>\n")

        print("<<< START ROUTE-AGRIGATE >>>")
        # Small modification from the original paper, was:
        # out = out + feature
        print("IN-1 {} IN-2 {}".format(out.shape, feature.shape))
        out = torch.cat((out, feature), dim=1)
        anlz_submod(torch.cat)
        print("GO {}".format(out.shape))
        print("<<< ENDED ROUTE-AGRIGATE >>>\n")
        print("GO {}".format(out.shape))
        print("<<< ENDED R_ASPP >>>\n")
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EfficientNet(nn.Module):
    #### INITIALIZATION OF THE EFFICIENTNET-LITE ENCODER
    ### We remove the classification head of the network, since we do not need it.
    ### For further details we refer to the backbone.py code.

    def __init__(self, backbone):
        super(EfficientNet, self).__init__()
        model = EffNet.from_name(backbone)
        
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc

        self.model = model

    def forward(self, x):
        print("<<< START EFFICIENTNET >>>")
        print("IN {}".format(x.shape))
        x = self.model._act(self.model._bn0(self.model._conv_stem(x)))
        [anlz_submod(s) for s in (self.model._conv_stem, self.model._bn0,self.model._act)]
        feature_maps = []
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            xin_shape = x.shape
            x = block(x, drop_connect_rate=drop_connect_rate)
            anlz_block(block,idx)
            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(x)
                if len(feature_maps)==4:break

        [sys.stdout.write("GO-{} {} ".format(no, fm.shape)) for no, fm in enumerate(feature_maps)]
        print("\n>>> ENDED EFFICIENTNET <<<\n")
        return feature_maps

class MalignancyDetector(nn.Module):
    #### INITIALIZATION OF THE WHOLE MODEL EFFICIENTNET-LITE + R-ASSP-LITE
    def __init__(self, backbone='efficientnet-lite1', num_classes=2, dropout=0.1):
        super(MalignancyDetector, self).__init__()

        ### DEFINE THE BACKBONE (ENCODER)
        self.base_forward = EfficientNet(backbone)

        ### COMPRESS THE FEATURES FROM THE BACKBONE
        self._conv_head = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(96)
            )

        ### INITIALIZE THE SEGMENTATION HEAD
        self.r_aspp = R_ASPP_module(in_channels_x=96, in_channels_f=40, num_classes=38)

        ### INITIALIZE THE FINAL CONVOLUTIONAL LAYER
        self.last_conv = nn.Sequential(
            nn.Dropout2d(dropout, False),
            nn.Conv2d(38, num_classes, kernel_size=1, stride=1)
        )

        self._act = nn.ReLU6()

        self._init_weight()

    def forward(self, x):
    #    print("Shape x    = {}".format(x.shape))
        #_, c2, _, c4 = self.base_forward(x)
        _1, c2, _3, c4 = self.base_forward(x)
        outR= self._act(self._conv_head(c4))

        out2= self.r_aspp(outR, c2)

        out3= self.last_conv(out2)

        mask = F.interpolate(out3, size=x.size()[2:], mode='bilinear', align_corners=True)

    #    print("Shape B._1 = {} c2 = {} _3 = {} c4 = {}".format(_1.shape, c2.shape, _3.shape, c4.shape))
    #    print("Shape outR = {} out2 = {} out3 = {}".format(outR.shape, out2.shape, out3.shape))
    #    print("Shape Mask = {}".format(mask.shape))
        return mask

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    ## If "python model.py" is run, this part of the code is executed, and tests the inference speed of the model.
    import time

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Define Parameters
    device = torch.device("cpu")
    imsize = 224

    # Load  trained Model
    model = MalignancyDetector(backbone='efficientnet-lite1', num_classes=2, dropout=0.).to(device)
    model = open_model(model, 'final.pth')
    model.eval()

    print ('Total trainable parameters: ', count_parameters(model))
    time_total = 0.

    #Test inference speed over 500 randomly generated pixels
    for idx in range(500):

        with torch.no_grad():
            time_start = time.time()
            input = torch.rand((1, 3, imsize, imsize), dtype=torch.float32, device=device)
            segmentation = model(input)

            time_stop = time.time()

        time_total += time_stop - time_start

    print('FPS: ', 500/time_total)


