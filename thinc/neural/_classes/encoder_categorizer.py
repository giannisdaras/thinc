# coding: utf8
from __future__ import unicode_literals, print_function
from .model import Model
from ...api import chain, clone, with_getitem, wrap, with_reshape
from .softmax import Softmax
from .affine import Affine
from .multiheaded_attention import MultiHeadedAttention
from .encoder_decoder import EncoderLayer, PytorchLayerNorm
from thinc.extra.wrappers import PyTorchWrapper
import torch
import torch.nn as nn


class Categorizer(Model):
    def __init__(self, nS=12, nM=768, nH=12, nO=2, device='cpu'):
        Model.__init__(self)
        self.nM = nM
        self.nO = nO
        self.nS = nS
        self.enc = clone(EncoderLayer(nM=nM, nH=nH, device=device), nS)
        self.affine = Affine(nI=nM, nO=nM)
        self.softmax = Softmax(nI=nM, nO=nO)
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM=nM, device=device))
        self.slicer = PyTorchWrapper(PytorchSlicer())
        self.device = device
        self.layers_ = [self.enc]

    def begin_update(self, inputs, drop=0.0):
        X0, Xmask = inputs
        (X1, _,), b_X1 = self.enc.begin_update((X0, Xmask))
        X2, b_X2 = self.norm.begin_update(X1)
        X3, b_X3 = self.slicer.begin_update(X2)
        X4, b_X4 = self.affine.begin_update(X3)
        X5, b_X5 = self.softmax.begin_update(X4)

        def finish_update(dX5, sgd=None):
            dX4 = b_X5(dX5, sgd=sgd)
            dX3 = b_X4(dX4, sgd=sgd)
            dX2 = b_X3(dX3)
            dX1 = b_X2(dX2, sgd=sgd)
            dX0 = b_X1(dX1, sgd=sgd)
            return dX0
        return X5, finish_update


class PytorchSlicer(nn.Module):
    def __init__(self):
        super(PytorchSlicer, self).__init__()

    def forward(self, x):
        return x[:, 0, :]
