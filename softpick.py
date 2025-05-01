#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-05-02 00:45:09 Friday

@author: Nikhil Kapila
"""

import torch
import torch.nn.functional as F

def softpick(x:torch.Tensor, dim:int=-1, eps:float=1e-6)->torch.Tensor:
    """Numerically safe version of softpick function
    ReLU(e^(x_i-m) - e^(-m)) / summation_1_N |e^(x_j-m) - e^(-m)| + eps
    Paper link: https://www.arxiv.org/pdf/2504.20966

    Args:
        x (torch.Tensor): Input tensor.
        dim (int, optional): Dimension along which softpick should be applied. Defaults to -1.
        eps (float, optional): Softpick epsilon used in Table 1 of the paper. Defaults to 1e-6.

    Returns:
        torch.Tensor: Output of softpick operation.
    """

    # m is the maximum value inside x 
    # https://pytorch.org/docs/stable/generated/torch.max.html
    # Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor in the given dimension dim. 
    # And indices is the index location of each maximum value found (argmax).
    mmax = torch.max(x, dim=dim, keepdim=True)
    m = mmax.values

    # numerator
    num = torch.exp(x-m) - torch.exp(-m)
    numer = F.relu(num)

    # denominator
    # https://pytorch.org/docs/stable/generated/torch.sum.html
    d = torch.abs(num) # |e^(x_j-m) - e^(-m)|

    # sum abs value + eps
    denom = torch.sum(d, dim, keepdim=True) + eps

    return numer/denom
