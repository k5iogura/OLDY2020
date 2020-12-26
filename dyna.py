import os,sys

import torch.nn as nn
import torch
import torch.nn.functional as F
from pdb import set_trace
from inspect import getmembers

class com():
    def __init__(self):
        self.comment = []
    def push(self,c):
        self.comment.append(c)
    def pop(self):
        return self.comment.pop()
    def str(self):
        rv=''
        for c in reversed(self.comment):
            rv+=str(c)+'.'
        rv='"'+rv+'"'
        return str(rv)
comment = com()

def hook(m, i, o):
    (name, in_channels, out_channels, kernel_size, stride ,padding, eps)=anlz_submod(m)
    if in_channels  is None and len(i[0].shape) >= 2: in_channels  = i[0].shape[1]
    if out_channels is None and len(o.shape)    >= 2: out_channels = o.shape[1]
    print(" ** {} {} {} {} {} {} {} {} {}".format(name,i[0].shape[-1],o.shape[-1],in_channels,out_channels,kernel_size,stride,padding,comment.str()))

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
#    if in_channels or 'function' in str(type(submod)):
#        print("{} {} {} {} {} {} {}".format(name, in_channels, out_channels, kernel_size, stride ,padding, eps))
#    else:
#        print("{}".format(submod))
    return (name, in_channels, out_channels, kernel_size, stride ,padding, eps)

def anlz_block_(block, out=None, no=""):
    block_name = block._get_name()
    no = "" if no == "" else "-"+str(no)
    print("{}{}".format(block_name,no))
    modules = block.__dict__['_modules']
    for k in modules.keys():
        submod = modules[k]
        #print("{}".format(submod))  #
        sys.stdout.write(' ')
        (name, in_channels, out_channels, kernel_size, stride ,padding, eps)=anlz_submod(submod)
    pass

class anlz_interpolate():
    def __init__(self, intensor=None):
        if intensor is not None:
            self.intensor_shape = intensor.shape

    def info(self, gotensor, size=None, mode=None, align_corners=False):
        print(" *** interpolate {} {} {} {} {}".format(gotensor.shape, size, mode, align_corners,comment.str()))

class anlz_cat():
    def __init__(self, intensor1, intensor2):
        self.intensor1_shape = intensor1.shape
        self.intensor2_shape = intensor2.shape

    def info(self, gotensor):
        print(" *** torch.cat {} {} {} {}".format(self.intensor1_shape, self.intensor2_shape, gotensor.shape,comment.str()))

