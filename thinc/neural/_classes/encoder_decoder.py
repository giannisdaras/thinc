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
from .multiheaded_attention import MultiHeadedAttention, PytorchMultiHeadedAttention
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
        self.dec = clone(DecoderLayer(nM=nM, nH=nH), nS)
        self.norm = PyTorchWrapper(PytorchLayerNorm(nM=300))

        # dec_i_grad = [1, 1, 0, 0]
        # dec_o_xp = None
        # dec_b_map = [[0], [1]]
        # dec_ret_x = [0, 1]
        # self.dec = PyTorchWrapper(Decoder(nM=nM, nH=nH, nS=nS), conf=[dec_i_grad, dec_o_xp, dec_b_map, dec_ret_x])
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
        sentX = None
        sentY = None
        (X1, _,), backprop_encode = self.enc.begin_update((X0, Xmask))
        X2, b_X2 = self.norm.begin_update(X1)
        (Y1, _, _, _), backprop_decode = self.dec.begin_update((Y0, X2, Xmask, Ymask))
        Y2, b_Y2 = self.norm.begin_update(Y1)
        # Y1, backprop_decode = self.dec.begin_update((Y0, X1, Xmask, Ymask))
        word_probs, backprop_output = self.proj.begin_update(Y2, drop=drop)

        def finish_update(d_word_probs, sgd=None):
            dY2 = backprop_output(d_word_probs, sgd=sgd)
            dY1 = b_Y2(dY2)
            zeros = Model.ops.xp.zeros(X0.shape, dtype=Model.ops.xp.float32)
            dY0, dX2 = backprop_decode((dY1, zeros), sgd=sgd)
            dX1 = b_X2(dX2)
            dX0 = backprop_encode(dX1, sgd=sgd)
            return (dX0, dY0)
        return (word_probs, Xmask), finish_update


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


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




class PytorchSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, nM=300, dropout=0.0):
        super(PytorchSublayerConnection, self).__init__()
        self.norm = PytorchLayerNorm(nM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(Model):
    def __init__(self, nM=300, nH=6, dropout=0.0):
        Model.__init__(self)
        self.attn = MultiHeadedAttention(nM=nM, nH=nH)
        self.ffd = PositionwiseFeedForward(nM, nM)
        self.nM = nM

    def begin_update(self, input, drop=0.0):
        X0, mask = input
        # TODO: the following two layers should do x + layer(norm(x))
        (X1, _), b_X1 = self.attn.begin_update((X0, mask, None))
        X2, b_X2 = self.ffd.begin_update(X1)
        def finish_update(dX2, sgd=None):
            dX1 = b_X2(dX2)
            dX0 = b_X1(X1)
            return X0
        return (X2, mask), finish_update


class Decoder(nn.Module):
    def __init__(self, nS=6, nH=6, nM=300):
        super(Decoder, self).__init__()
        layer = PytorchDecoderLayer()
        self.layers = clones(layer, nS)
        self.norm = PytorchLayerNorm(nM)

    def forward(self, input):
        x, memory, src_mask, tgt_mask = input
        for layer in self.layers:
            x = layer((x, memory, src_mask, tgt_mask))
        return self.norm(x)


class PytorchDecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, nM=300, nH=6, dropout=0.0):
        super(PytorchDecoderLayer, self).__init__()
        self.nM = nM
        self.self_attn = PytorchMultiHeadedAttention(nM=nM, nH=nH)
        self.src_attn = PytorchMultiHeadedAttention(nM=nM, nH=nH)
        self.feed_forward = PytorchPositionwiseFeedForward(nM, nM)
        self.sublayer = clones(PytorchSublayerConnection(nM, dropout), 3)

    def forward(self, input):
        x, memory, src_mask, tgt_mask = input
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn((x, x, x, tgt_mask)))
        x = self.sublayer[1](x, lambda x: self.src_attn((x, m, m, src_mask)))
        return self.sublayer[2](x, self.feed_forward)


class DecoderLayer(Model):
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        # TODO: residual
        self.y_attn = MultiHeadedAttention(nM=nM, nH=nH)
        # TODO: residual
        self.x_attn = MultiHeadedAttention(nM=nM, nH=nH)
        # outer attention config
        # o_xp = None
        # i_grad = [1, 1, 0, 0]
        # b_map = [[0, 1]]
        # ret_x = [0, 1]
        # conf = [i_grad, o_xp, b_map, ret_x]
        # self.x_attn = PyTorchWrapper(PytorchMultiHeadedAttention(nM=nM, nH=nH), conf=conf)
        self.ffd = PositionwiseFeedForward(nM, nM)

    def begin_update(self, input, drop=0.0):
        Y0, X0, X_mask, Y_mask = input
        (Y1, _), b_Y1 = self.y_attn.begin_update((Y0, Y_mask, None))
        # Y2, b_Y2 = self.x_attn.begin_update((Y1, X0, X0, X_mask))
        (Y2, _), b_Y2 = self.x_attn.begin_update((Y1, X0, X_mask, None, None))
        Y3, b_Y3 = self.ffd.begin_update(Y2)

        def finish_update(dY3_dX, sgd=None):
            dY3, dX = dY3_dX
            dY2 = b_Y3(dY3)
            dY1, dX0 = b_Y2(dY2)
            dY0 = b_Y1(dY1)
            dX0 += dX
            return dY0, dX0
        return (Y3, X0, X_mask, Y_mask), finish_update


class PytorchPositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, nM=300, nO=300, dropout=0.1):
        super(PytorchPositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(nM, nO)
        self.w_2 = nn.Linear(nO, nM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionwiseFeedForward(Model):
    def __init__(self, nM=300, nO=300, dropout=0.0):
        Model.__init__(self)
        self.ffd1 = with_reshape(ReLu(nM, nO))
        self.ffd2 = with_reshape(Affine(nO, nM))
        self.layers_ = [self.ffd1, self.ffd2]
        self.nO = nO

    def begin_update(self, X0, drop=0.0):
        X1, b_ffd1 = self.ffd1.begin_update(X0)
        X2, b_ffd2 = self.ffd2.begin_update(X1)

        def finish_update(dX2):
            dX1 = b_ffd2(dX2)
            dX0 = b_ffd1(dX1)
            return dX0
        return X2, finish_update
