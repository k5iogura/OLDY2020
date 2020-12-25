import os,sys

import torch.nn as nn
import torch
import torch.nn.functional as F
from pdb import set_trace
from inspect import getmembers

def hook(m, i, o):
    sys.stdout.write("*********** ")
    #anlz_block(m)
#    print(m._get_name(),i[0].shape,o[0].shape)
    pass

def set_hook(net):
    for name, layer in net._modules.items():
        if isinstance(layer, nn.Sequential):
            pass
        else:
            layer.register_forward_hook(hook)
            set_hook(layer)

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
    if in_channels or 'function' in str(type(submod)):
        print("{} {} {} {} {} {} {}".format(name, in_channels, out_channels, kernel_size, stride ,padding, eps))
    else:
        print("{}".format(submod))
    return (name, in_channels, out_channels, kernel_size, stride ,padding, eps)

def anlz_block(block, out=None, no=""):
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

