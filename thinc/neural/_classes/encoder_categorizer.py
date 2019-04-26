# coding: utf8
from __future__ import unicode_literals, print_function
from .model import Model
from ...api import chain, clone, with_getitem, wrap, with_reshape
from .softmax import Softmax
from .affine import Affine
from .multiheaded_attention import MultiHeadedAttention
from .encoder_decoder import EncoderLayer
from thinc.extra.wrappers import PyTorchWrapper
import torch
import torch.nn as nn


class Categorizer(Model):
    def __init__(self, nS=12, nM=768, nH=12, nO=2):
        Model.__init__(self)
        self.nM = nM
        self.nO = nO
        self.nS = nS
        self.enc = clone(EncoderLayer(nM=nM, nH=nH), nS)
        self.softmax = with_reshape(Softmax(nI=nM, nO=nO))
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM=nM))
        self.layers_ = [self.enc]

    def begin_update(self, inputs, drop=0.0):
        X0, Xmask = inputs
        (X1, _,), b_X1 = self.enc.begin_update((X0, Xmask))
        X2, b_X2 = self.norm.begin_update(X1)
        X3, b_X3 = self.softmax.begin_update(X2)
        X4 = X3[:, 0, :]

        def finish_update(dX4, sgd=None):
            dX3 = Model.ops.xp.zeros((X3.shape))
            dX3[:, 0, :] = dX4
            dX2 = b_X3(dX3, sgd=sgd)
            dX1 = b_X2(dX2, sgd=sgd)
            dX0 = b_X1(dX1, sgd=sgd)
            return dX0
        return X4, finish_update


class PytorchLayerNorm(nn.Module):
    def __init__(self, nM=300, eps=1e-6):
        super(PytorchLayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(nM))
        self.b_2 = nn.Parameter(torch.zeros(nM))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
