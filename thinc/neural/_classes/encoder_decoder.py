# coding: utf8
from __future__ import unicode_literals, print_function
from .model import Model
from ...api import chain, clone, with_getitem, wrap, with_reshape
from .softmax import Softmax
from .relu import ReLu
from .layernorm import LayerNorm
from .maxout import Maxout
from .resnet import Residual
from .affine import Affine
from .multiheaded_attention import MultiHeadedAttention
from .positionwise_ffd import PositionwiseFeedForward
import copy
import math
import numpy as np
import torch.nn as nn
import torch
from thinc.extra.wrappers import PyTorchWrapper
import torch.nn.functional as F


class EncoderDecoder(Model):
    def __init__(self, nS=1, nH=6, nM=300, nTGT=10000):
        '''
        EncoderDecoder consists of an encoder stack, a decoder stack and an
        output layer which is a linear + softmax.
        Parameters explanation:
            nS: the number of encoders/decoders in the stack
            nH: the number of heads in the multiheaded attention
            nM: the token's embedding size
            nTGT: the number of unique words in output vocabulary
        '''
        Model.__init__(self)
        self.nS = nS
        self.nH = nH
        self.nM = nM
        self.nTGT = nTGT
        self.enc = clone(EncoderLayer(nM=nM, nH=nH), nS)
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM=nM))
        self.dec = clone(DecoderLayer(nM=nM, nH=nH), nS)
        self.proj = with_reshape(Softmax(nO=nTGT, nI=nM))
        self._layers = [self.enc, self.dec, self.proj]

    def begin_update(self, inputs, drop=0.0):
        '''
        A batch object flows through the network. It contains input, output and
        corresponding masks. Input changes while the object travels through
        the network. Output is the golden output.
        Input: nB x nL x nM
        '''
        X0, Xmask, Y0, Ymask = inputs
        (X1, _,), backprop_encode = self.enc.begin_update((X0, Xmask))
        X2, b_X2 = self.norm.begin_update(X1)
        (Y1, _, _, _), backprop_decode = self.dec.begin_update((Y0, X2, Xmask, Ymask))
        Y2, b_Y2 = self.norm.begin_update(Y1)
        word_probs, backprop_output = self.proj.begin_update(Y2, drop=drop)

        def finish_update(d_word_probs, sgd=None):
            dY2 = backprop_output(d_word_probs, sgd=sgd)
            dY1 = b_Y2(dY2, sgd=sgd)
            zeros = Model.ops.xp.zeros(X0.shape, dtype=Model.ops.xp.float32)
            dY0, dX2 = backprop_decode((dY1, zeros), sgd=sgd)
            dX1 = b_X2(dX2, sgd=sgd)
            dX0 = backprop_encode(dX1, sgd=sgd)
            return (dX0, dY0)
        return (word_probs, Xmask), finish_update


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


class EncoderLayer(Model):
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        self.attn = MultiHeadedAttention(nM=nM, nH=nH)
        self.ffd = PositionwiseFeedForward(nM, nM)
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM))
        self.nM = nM
        self.layers_ = [self.attn, self.ffd, self.norm]

    def begin_update(self, input, drop=0.0):
        X0, mask = input
        X1, b_X1 = self.attn.begin_update((X0, mask, None))
        X2, b_X2 = self.norm.begin_update(X1)
        X3 = X0 + X2

        X4, b_X4 = self.ffd.begin_update(X3)
        X5, b_X5 = self.norm.begin_update(X4)
        X6 = X3 + X5

        def finish_update(dX6, sgd=None):
            dX5 = dX6
            dX4 = b_X5(dX5, sgd=sgd)
            dX3 = b_X4(dX4, sgd=sgd)
            dX3 += dX6

            dX2 = dX3
            dX1 = b_X2(dX2, sgd=sgd)
            dX0 = b_X1(dX1, sgd=sgd)

            dX0 += dX3
            return X0
        return (X6, mask), finish_update


class DecoderLayer(Model):
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        self.y_attn = MultiHeadedAttention(nM=nM, nH=nH)
        self.x_attn = MultiHeadedAttention(nM=nM, nH=nH)
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM))
        self.ffd = PositionwiseFeedForward(nM, nM)
        self.layers_ = [self.norm, self.y_attn, self.x_attn, self.ffd]

    def begin_update(self, input, drop=0.0):
        Y0, X0, X_mask, Y_mask = input
        Y1, b_Y1 = self.y_attn.begin_update((Y0, Y_mask, None))
        Y2, b_Y2 = self.norm.begin_update(Y1)
        Y3 = Y0 + Y2
        Y4, b_Y4 = self.x_attn.begin_update((Y3, X0, X_mask, None, None))
        Y5, b_Y5 = self.norm.begin_update(Y4)
        Y6 = Y3 + Y5
        Y7, b_Y7 = self.ffd.begin_update(Y6)

        def finish_update(dI, sgd=None):
            dY7, dX = dI
            dY6 = b_Y7(dY7, sgd=sgd)
            dY5 = dY6
            dY4 = b_Y5(dY5, sgd=sgd)
            dY3, dX0 = b_Y4(dY4, sgd=sgd)
            dY3 += dY6
            dY2 = dY3
            dY1 = b_Y2(dY2, sgd=sgd)
            dY0 = b_Y1(dY1, sgd=sgd)
            dY0 += dY3
            dX0 += dX
            return (dY0, dX0)
        return (Y7, X0, X_mask, Y_mask), finish_update
