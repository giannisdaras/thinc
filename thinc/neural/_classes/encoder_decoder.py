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
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        # self.x_attn = MultiHeadedAttention(nM, nH, layer='Decoder')
        # self.y_attn = MultiHeadedAttention(nM, nH, layer='Decoder')
        self.x_attn = PyTorchWrapper(PytorchMultiHeadedAttention(nM, nH, layer='Decoder'), conf=[[True, True, False, False, False], [True, False], [1, 1, 0, 0], [1, 1]])
        self.y_attn = PyTorchWrapper(PytorchMultiHeadedAttention(nM, nH, layer='Decoder'), conf=[[True, False, False], [True, False], None, [1, 1]])
        self.ffd = with_reshape(LayerNorm(Affine(nM, nM)))
        self._layers = [self.x_attn, self.y_attn, self.ffd]

    def begin_update(self, X_Y, drop=0.0):
        (X0, Xmask, sentX), (Y0, Ymask, sentY) = X_Y
        (Y1, _), bp_self_attn = self.y_attn.begin_update((Y0, Ymask, sentY))
        (mixed, _), bp_mix_attn = self.x_attn.begin_update((Y1, X0, Xmask, sentY, sentX))
        output, bp_output = self.ffd.begin_update(mixed)

        def finish_update(dXprev_d_output, sgd=None):
            dXprev, d_output = dXprev_d_output
            d_mixed = bp_output(d_output, sgd=sgd)
            dY1, dX0 = bp_mix_attn(d_mixed, sgd=sgd)
            dY0 = bp_self_attn(dY1, sgd=sgd)
            return (dX0 + dXprev, dY0)

        return ((X0, Xmask, sentX), (output, Ymask, sentY)), finish_update


class PoolingDecoder(Model):
    def __init__(self, nM):
        Model.__init__(self)
        self.nM = nM
        self.ffd = LayerNorm(Maxout(nO=nM, nI=nM*3, pieces=3))
        self._layers = [self.ffd]

    def begin_update(self, X_Y, drop=0.):
        (X0, Xmask, _), (Y0, Ymask, _) = X_Y
        X_masked = self.ops.xp.copy(X0)
        X_masked[Xmask[:, 0, :] == 0] = -math.inf
        Xpool = X_masked.max(axis=1, keepdims=True)

        Y_masked = self.ops.xp.copy(Y0)
        Y_masked[Ymask[:, -1, :] == 0] = -math.inf
        Ypool = self.ops.allocate((X0.shape[0], X0.shape[1], self.nM))

        # maxing only over previous elements
        for i in range(X0.shape[1]):
            Ypool[:, i, :] = Y_masked[:, :i+1, :].max(axis=1, keepdims=True).squeeze()

        mixed = self.ops.allocate((X0.shape[0], X0.shape[1], self.nM*3))
        mixed[:, :, :self.nM] = Xpool
        mixed[:, :, self.nM:self.nM*2] = Ypool
        mixed[:, :, self.nM*2:] = Y0
        output, bp_output = self.ffd.begin_update(mixed.reshape((-1, self.nM*3)))
        output = output.reshape((X0.shape[0], X0.shape[1], output.shape[1]))

        def backprop_pooling_decoder(dX_d_output, sgd=None):
            dXin, d_output = dX_d_output
            d_output = d_output.reshape((X0.shape[0]*X0.shape[1], d_output.shape[2]))
            d_mixed = bp_output(d_output, sgd=sgd)
            d_mixed = d_mixed.reshape((X0.shape[0], X0.shape[1], d_mixed.shape[1]))
            dXpool = d_mixed[:, :, :self.nM]
            dYpool = d_mixed[:, :, self.nM:self.nM*2]
            dY0 = d_mixed[:, :, self.nM*2:]
            dX0 = self.ops.allocate(X0.shape)
            for i in range(X0.shape[0]):
                for j in range(X0.shape[1]):
                    for k in range(X0.shape[2]):
                        if X0[i, j, k] >= Xpool[i, 0, k]:
                            dX0[i, j, k] += dXpool[i, j, k]
            for i in range(Y0.shape[0]):
                for j in range(Y0.shape[1]):
                    for k in range(Y0.shape[2]):
                        if Y0[i, j, k] >= Ypool[i, j, k]:
                            dY0[i, j, k] += dYpool[i, j, k]
            return dXin + dX0, dY0
        return ((X0, Xmask), (output, Ymask)), backprop_pooling_decoder


class PytorchDecoder(nn.Module):
    def __init__(self, nH, nM):
        self.nH = nH
        self.nM = nM
        self.x_attn = PytorchMultiHeadedAttention(nM, nH, layer='Decoder')
        self.y_attn = PytorchMultiHeadedAttention(nM, nH, layer='Decoder')
        self.ffd = nn.Linear(nM, nM)

    def forward(self, X_Y):
        (X0, Xmask, sentX), (Y0, Ymask, sentY) = X_Y
        (Y1, _) = self.y_attn((Y0, Ymask, sentY))
        (mixed, _) = self.x_attn((Y1, X0, Xmask, sentY, sentX))
        output = self.ffd(mixed)
        return ((X0, Xmask), (output, Ymask))
