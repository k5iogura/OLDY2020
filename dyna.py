import os,sys

import torch.nn as nn
import torch
import torch.nn.functional as F
import csv
from pdb import set_trace
from inspect import getmembers

# CSV
fout = sys.stdout
fout = open('maglinancy.csv','w')
writer = csv.writer(fout)
writer.writerow(['Unit','K','N','M','S','Fi','Fo','P','Comment'])

def outKNMSFiFo(s):
    name,K,N,M,S,Fi,Fo,P,comstr=s
    name="" if name is None else name
    K="" if K is None else K
    N="" if N is None else N
    M="" if M is None else M
    S="" if S is None else S
    Fi="" if Fi is None else Fi
    Fo="" if Fo is None else Fo
    P="" if P is None else P
    print("{} {} {} {} {} {} {} {} {}".format(name,K,N,M,S,Fi,Fo,P,comstr))
    writer.writerow([name,K,N,M,S,Fi,Fo,P,comstr])

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
    outKNMSFiFo([None,None,None,None,None,None,None,None,comstr])

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

# KNMSFiFo
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
    outKNMSFiFo((name,K,N,M,S,Fi,Fo,P,comstr))

def set_hook(net):
    assert isinstance(net, nn.Sequential),"dont use this others of nn.Sequential {}".format(type(net))
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            pass
        else:
            layer.register_forward_hook(hook)
            #set_hook(layer)

def anlz_submod(submod, out=None):
    class_name = in_channels = out_channels = kernel_size = stride = padding = eps = None
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
    name = submod.__name__ if 'function' in str(type(submod)) else name
    return (name, in_channels, out_channels, kernel_size, stride ,padding, eps)

#def anlz_block_(block, out=None, no=""):
    #block_name = block._get_name()
    #no = "" if no == "" else "-"+str(no)
    #print("{}{}".format(block_name,no))
    #modules = block.__dict__['_modules']
    #for k in modules.keys():
        #submod = modules[k]
        #sys.stdout.write(' ')
        #(name, in_channels, out_channels, kernel_size, stride ,padding, eps)=anlz_submod(submod)
    #pass

class anlz_product():
    def __init__(self, intensor1):
        self.intensor1_shape=intensor1.shape
    def info(self, gotensor, intensor2):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P=None
        comstr=comment.str(append="NUMPY PRODUCT IN1 AND IN2")
        outKNMSFiFo(("(ProductOperator)",K,N,M,S,Fi,Fo,P,comstr))

class anlz_plus():
    def __init__(self, intensor1):
        self.intensor1_shape=intensor1.shape
    def info(self, gotensor, intensor2):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P=None
        comstr=comment.str(append="SKIPADD CURRENT TENSOR AND BLOCK's INPUT")
        outKNMSFiFo(("(PlusOperator)",K,N,M,S,Fi,Fo,P,comstr))

#KNMSFiFo
class anlz_interpolate():
    def __init__(self, intensor=None):
        intensor_shape = None
        if intensor is not None: self.intensor_shape = intensor.shape

    def info(self, gotensor, size=None, mode=None, align_corners=False):
        gotensor_shape = gotensor.shape # N,C,H,W
        K =size[-1] if size is not None and (isinstance(size,tuple) or isinstance(size,list)) else size
        N =self.intensor_shape[1] if self.intensor_shape is not None else None
        M =     gotensor_shape[1] if      gotensor_shape is not None else None
        S=None
        Fi=self.intensor_shape[-1] if self.intensor_shape is not None else None
        Fo=     gotensor_shape[-1] if      gotensor_shape is not None else None
        P = None
        size = [i for i in size] if size is not None else size
        comstr=comment.str(append=" size="+str(size)+" mode="+mode+" align_corners="+str(align_corners))
        outKNMSFiFo(("F.interpolate",K,N,M,S,Fi,Fo,P,comstr))

class anlz_cat():
    def __init__(self, intensor1, intensor2):
        self.intensor1_shape = intensor1.shape
        self.intensor2_shape = intensor2.shape
        assert self.intensor1_shape == self.intensor2_shape

    def info(self, gotensor):
        K=None
        N=self.intensor1_shape[1]
        M=     gotensor.shape[1]
        S=None
        Fi=self.intensor1_shape[-1]
        Fo=     gotensor.shape[-1]
        P = None
        comstr=comment.str()
        outKNMSFiFo(("torch.cat",K,N,M,S,Fi,Fo,P,comstr))

