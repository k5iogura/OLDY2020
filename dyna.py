import os,sys,re

import torch.nn as nn
import torch
import torch.nn.functional as F
import csv
import numpy as np
from pdb import set_trace
from inspect import getmembers

# CSV
fout = sys.stdout
fout = open('maglinancy.csv','w')
writer = csv.writer(fout)
# KNMSFiFo : KernelSize In-Channels Out-Channels Stride InFeatureSize OutFeatureSize and Padding, Comment
writer.writerow(['Unit','K','N','M','S','Fi','Fo','P','Ti1','Ti2','To','Comment'])

# For unique id of Tensor
__baseNN=np.power(10,8)
def uid(idx): return id(idx)%__baseNN

def outKNMSFiFo(s):
    name,K,N,M,S,Fi,Fo,P,Ti1,Ti2,To,comstr=s
    name="" if name is None else name
    K="" if K is None else K
    N="" if N is None else N
    M="" if M is None else M
    S="" if S is None else S
    Fi="" if Fi is None else Fi
    Fo="" if Fo is None else Fo
    P="" if P is None else P
    Ti1="" if Ti1 is None else Ti1
    Ti2="" if Ti2 is None else Ti2
    To="" if To is None else To
    print(re.sub('^ *','',"{} {} {} {} {} {} {} {} {} {} {} {}".format(name,K,N,M,S,Fi,Fo,P,Ti1,Ti2,To,comstr)))
    writer.writerow([name,K,N,M,S,Fi,Fo,P,Ti1,Ti2,To,comstr])

def outComment(comstr,in1_tensor=None, in2_tensor=None):
    in1_shape = in2_shape = None
    if in1_tensor is not None: in1_shape = str([i for i in in1_tensor.shape])
    if in2_tensor is not None: in2_shape = str([i for i in in2_tensor.shape])
    if in1_tensor is not None:
        comstr += "IN " if in2_tensor is None else "IN1 "
        comstr += in1_shape
        if in2_tensor is not None:
            comstr += " IN2 "
            comstr += in2_shape
    outKNMSFiFo([None,None,None,None,None,None,None,None,None,None,None,comstr])

def outCommentEnd(comstr,go1_tensor=None, go2_tensor=None, go3_tensor=None, go4_tensor=None):
    go1_shape = go2_shape = go3_shape = go4_shape = None
    if go1_tensor is not None: go1_shape = str([i for i in go1_tensor.shape])
    if go2_tensor is not None: go2_shape = str([i for i in go2_tensor.shape])
    if go3_tensor is not None: go3_shape = str([i for i in go3_tensor.shape])
    if go4_tensor is not None: go4_shape = str([i for i in go4_tensor.shape])
    if go1_tensor is not None:
        comstr += "GO " if go2_tensor is None else "GO1 "
        comstr += go1_shape
        if go2_tensor is not None:
            comstr += " GO2 "
            comstr += go2_shape
            if go3_tensor is not None:
                comstr += " GO3 "
                comstr += go3_shape
                if go4_tensor is not None:
                    comstr += " GO4 "
                    comstr += go4_shape
    outKNMSFiFo([None,None,None,None,None,None,None,None,None,None,None,comstr])

class com():
    def __init__(self):
        self.comment = []
    def push(self,c):
        self.comment.append(c)
    def pop(self):
        return self.comment.pop()
    def str(self,append=""):
        rv=''
        for c in reversed(self.comment):
            rv+=str(c)+'.'
        #rv='"'+rv+str(append)+'"'
        rv=rv+str(append)
        return str(rv)
comment = com()

def hook(m, i, o):
    (name, in_channels, out_channels, kernel_size, stride ,padding, eps)=anlz_submod(m)
    if in_channels  is None and len(i[0].shape) >= 2: in_channels  = i[0].shape[1]
    if out_channels is None and len(o.shape)    >= 2: out_channels = o.shape[1]
    K=kernel_size[-1] if isinstance(kernel_size,tuple) or isinstance(kernel_size,list) else kernel_size
    N=in_channels
    M=out_channels
    Fi=i[0].shape[-1]
    Fo=o.shape[-1]
    S=stride[-1] if isinstance(stride,tuple) or isinstance(stride,list) else stride
    P=padding
    comstr=comment.str()
    Ti1=uid(i[0]) if len(i)>=1 else None
    Ti2=uid(i[1]) if len(i)>=2 else None
    To=uid(o) if o is not None else None
    outKNMSFiFo((name,K,N,M,S,Fi,Fo,P,Ti1,Ti2,To,comstr))
    assert len(i) <= 2, "{} inputs are not supported".format(len(i))

def set_hook(net):
    assert isinstance(net, nn.Sequential),"dont use this others of nn.Sequential {}".format(type(net))
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            pass
        else:
            layer.register_forward_hook(hook)

def anlz_submod(submod, out=None):
    class_name = in_channels = out_channels = kernel_size = stride = padding = eps = bias = None
    try:
        name = submod._get_name()
    except:
        name = submod.__class__
    try:
        class_name = submod.__class__
    except:
        pass
    try:
        in_channels = submod.in_channels
    except:
        pass
    try:
        out_channels = submod.out_channels
    except:
        pass
    try:
        kernel_size = submod.kernel_size
    except:
        pass
    try:
        stride = submod.stride
    except:
        pass
    try:
        padding = submod.static_padding.padding
    except:
        try:
            padding = submod.padding
        except:
            pass
    try:
        eps = submod.eps
    except:
        pass
    try:
        bias = submod.bias
    except:
        pass

    if bias is not None and class_name != torch.nn.modules.batchnorm.BatchNorm2d:
        print("\t\tBIAS {} {}".format(class_name,bias.shape))
    name = submod.__name__ if 'function' in str(type(submod)) else name
    return (name, in_channels, out_channels, kernel_size, stride ,padding, eps)

class anlz_product():
    def __init__(self, intensor1):
        self.intensor1_shape=intensor1.shape
        self.Ti1 = uid(intensor1)
    def info(self, gotensor, intensor2):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P=None
        Ti2=uid(intensor2)
        To=uid(gotensor)
        comstr=comment.str(append="NUMPY PRODUCT")
        outKNMSFiFo(("(ProductOperator)",K,N,M,S,Fi,Fo,P,self.Ti1,Ti2,To,comstr))

class anlz_plus():
    def __init__(self, intensor1):
        self.intensor1_shape=intensor1.shape
        self.Ti1 = uid(intensor1)
    def info(self, gotensor, intensor2):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P=None
        #comstr=comment.str(append="SKIPADD CURRENT TENSOR AND BLOCK's INPUT")
        Ti2=uid(intensor2)
        To=uid(gotensor)
        comstr=comment.str(append=" SKIPADD")
        outKNMSFiFo(("(PlusOperator)",K,N,M,S,Fi,Fo,P,self.Ti1,Ti2,To,comstr))

#KNMSFiFo
class anlz_interpolate():
    def __init__(self, intensor=None):
        intensor_shape = None
        if intensor is not None: self.intensor_shape = intensor.shape
        self.Ti1=uid(intensor) if intensor is not None else None

    def info(self, gotensor, size=None, mode=None, align_corners=False):
        gotensor_shape = gotensor.shape # N,C,H,W
        K =None
        N =self.intensor_shape[1] if self.intensor_shape is not None else None
        M =     gotensor_shape[1] if      gotensor_shape is not None else None
        S=None
        Fi=self.intensor_shape[-1] if self.intensor_shape is not None else None
        Fo=     gotensor_shape[-1] if      gotensor_shape is not None else None
        P = None
        size = [i for i in size] if size is not None else size
        comstr=comment.str(append=" size="+str(size)+" mode="+mode+" align_corners="+str(align_corners))
        Ti2=None
        To=uid(gotensor)
        outKNMSFiFo(("F.interpolate",K,N,M,S,Fi,Fo,P,self.Ti1,Ti2,To,comstr))

class anlz_cat():
    def __init__(self, intensor1, intensor2):
        self.intensor1_shape = intensor1.shape
        self.intensor2_shape = intensor2.shape
        assert self.intensor1_shape == self.intensor2_shape
        self.Ti1=uid(intensor1) if intensor1 is not None else None
        self.Ti2=uid(intensor2) if intensor2 is not None else None

    def info(self, gotensor):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P = None
        comstr=comment.str()
        To=uid(gotensor)
        outKNMSFiFo(("torch.cat",K,N,M,S,Fi,Fo,P,self.Ti1,self.Ti2,To,comstr))

