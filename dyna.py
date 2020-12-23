import os,sys

import torch.nn as nn
import torch
import torch.nn.functional as F
from pdb import set_trace
from inspect import getmembers

def anlz_submod(submod):
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
    print("{} {} {} {} {} {} {}".format(name, in_channels, out_channels, kernel_size, stride ,padding, eps))
    return (name, in_channels, out_channels, kernel_size, stride ,padding, eps)

def anlz_block(block, no=""):
    block_name = block._get_name()
    no = "" if no == "" else "-"+str(no)
    print("{}{}".format(block_name,no))
    modules = block.__dict__['_modules']
    for k in modules.keys():
        submod = modules[k]
        sys.stdout.write(' ')
        (name, in_channels, out_channels, kernel_size, stride ,padding, eps)=anlz_submod(submod)
    pass

