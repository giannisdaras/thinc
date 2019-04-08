from __future__ import unicode_literals, print_function

from .model import Model
from .affine import Affine
from ...api import with_reshape
from thinc.extra.visualizer import visualize_attention
from thinc.extra.wrappers import xp2torch
import numpy as np
import torch.nn as nn
import torch
import math

class MultiHeadedAttention(Model):
    ''' This class implements multiheaded attention. It can be used for self
    attention or outer attention, depending on our needs. There is no left
    and right context width. We attend to the whole sentence and we take
    care of the masks to adjust appropriately. There are no actual different
    weight matrices for each head, but a bigger weight matrix for all heads.
    Going to bigger dimensions is the key to get the multiple heads.
    For the time being; key, query and value matrices are supposed to have the
    same length.
    '''
    def __init__(self, nM=300, nH=6, layer='Encoder'):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.layer = layer
        self.get_queries = with_reshape(Affine(nM, nM))
        self.get_keys = with_reshape(Affine(nM, nM))
        self.get_values = with_reshape(Affine(nM, nM))
        self.get_output = with_reshape(Affine(nM, nM))
        self._layers = [self.get_queries, self.get_keys, self.get_values, self.get_output]

    def begin_update(self, input, drop=0.0):
        # Queries come from input[0], keys and values from input[1]
        if len(input) == 3:
            x0, mask, sentX = input
            sentY = sentX
            y0 = x0
            self_attention = True
        else:
            self_attention = False
            x0, y0, mask, sentX, sentY = input
        ''' Shapes '''
        # x0: nB, nL, nM
        # q0: nB, nL, nM
        # k0: nB, nL, nM
        # v0: nB, nL, nM
        # q1: nB, nL, nH, nD
        # k1: nB, nL, nH, nD
        # v1: nB, nL, nH, nD
        # x1: nB, nL, nH, nD
        # x2: nB, nL, nM
        # x3: nB, nL, nM
        nB, nL, nD, nH = x0.shape[0], x0.shape[1], self.nD, self.nH
        q0, get_dx0 = self.get_queries.begin_update(x0)
        q1 = q0.reshape(nB, -1, self.nH, self.nD)
        k0, get_dy0_1 = self.get_keys.begin_update(y0)
        k1 = k0.reshape(nB, -1, self.nH, self.nD)
        v0, get_dy0_2 = self.get_values.begin_update(y0)
        v1 = v0.reshape(nB, -1, self.nH, self.nD)

        x1, get_dq1_dk1_dv1 = self.attn(q1, k1, v1, mask=mask, sentX=sentX,
                                        sentY=sentY, self_attn=self_attention)

        x2 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2]*x1.shape[3])
        x3, get_dx2 = self.get_output.begin_update(x2)

        def finish_update(dx3, sgd=None):
            dx2 = get_dx2(dx3, sgd=sgd)
            dx1 = dx2.reshape(nB, nL, nH, nD)
            dq1, dk1, dv1 = get_dq1_dk1_dv1(dx1)
            dv0 = dv1.reshape(nB, nL, nH, nD)
            dk0 = dk1.reshape(nB, nL, nH, nD)
            dq0 = dq1.reshape(nB, nL, nH, nD)
            dy0 = get_dy0_2(dv0, sgd=sgd) + get_dy0_1(dk0, sgd=sgd)
            dx0 = get_dx0(dq0, sgd=sgd)
            if self_attention:
                return dx0 + dy0
            else:
                return (dx0, dy0)
        return (x3, mask), finish_update

    def attn(self, Q, K, V, mask=None, sentX=None, sentY=None, self_attn=True):
        '''
        Compute attention on (query, key, value) triplets.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        '''
        # query shape: nB, nL, nH, nD

        S0, bp_scaled_dp = self._scaled_dot_prod(Q, K)
        S1, bp_mask = self._mask(S0, mask)
        S2, bp_softmax = self._softmax(S1)
        if sentX is not None and sentY is not None:
            visualize_attention(sentX, sentY, S2[0], layer=self.layer,
            self_attn=self_attn)
        S3, bp_apply_attn = self._apply_attn(S2, V)

        def backprop_attn(dS3):
            ''' Attention three inputs, one output '''
            dS2, dV = bp_apply_attn(dS3)
            dS1 = bp_softmax(dS2)
            dS0 = bp_mask(dS1)
            dQ, dK = bp_scaled_dp(dS0)
            #print("attn", "dQ", dQ.mean(), dQ.var(), "dK", dK.mean(), dK.var(), "dV", dV.mean(), dV.var())
            return dQ, dK, dV
        return S3, backprop_attn

    def _scaled_dot_prod(self, Q0, K0):
        # nB: #Sentences, nL: #Length, nH: #Heads, nD: #Dimensions
        nB, nL, nH, nD = Q0.shape
        # Shape of Q0: (nB, nL, nH, nD)
        # Shape of K0: (nB, nL, nH, nD)
        # --> (nB*nH, nL, nD)

        Q1 = Q0.transpose(0, 2, 1, 3).reshape(nB*nH, nL, nD)

        # --> (nB*nH, nD, nL)
        K1 = K0.transpose(0, 2, 3, 1).reshape(nB*nH, nD, nL)

        K2 = K1 / self.ops.xp.sqrt(self.nM)
        # (nB*nH, nL, nD) @ (nB*nH, nD, nL) --> (nB*nH, nL, nL)

        S = self.ops.xp.matmul(Q1, K2)

        def backprop_attn1(dS):
            # (nB*nH, nL, nL) @ (nB*nH, nD, nL).T --> (nB*nH, nL, nD)
            dS = dS.reshape(nB*nH, nL, nL)
            dQ1 = self.ops.xp.matmul(dS, K2.transpose(0, 2, 1))
            # (nB*nH, nL, nD).T @ (nB*nH, nL, nL) --> (nB*nH, nD, nL)
            dK2 = self.ops.xp.matmul(Q1.transpose(0, 2, 1), dS)
            dK1 = dK2 / self.ops.xp.sqrt(self.nM)
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 2, 3, 1)
            dQ0 = dQ1.reshape((nB, nH, nL, nD)).transpose(0, 2, 1, 3)
            return dQ0, dK0
        return S.reshape((nB, nH, nL, nL)), backprop_attn1

    def _mask(self, S0, mask):
        S1 = S0.transpose(1, 0, 2, 3)
        S2 = S1 * mask - (1 - mask) * (1e9)
        S3 = S2.transpose(1, 0, 2, 3)

        def backprop_attn2(dS3):
            dS2 = dS3.transpose(1, 0, 2, 3)
            dS1 = dS2 * mask
            dS0 = dS1.transpose(1, 0, 2, 3)
            return dS0

        return S3, backprop_attn2

    def _softmax(self, S0):
        ''' A simple softmax to the scores '''
        # S0: nB, nH, nL, nL
        # S1: nB, nH, nL, nL
        S1 = self.ops.softmax(S0)

        def backprop_attn3(dS1):
            dS0 = dS1 * S1
            sum_dS0 = dS0.sum(axis=2, keepdims=True)
            dS0 -= sum_dS0 * S1
            return dS0
        return S1, backprop_attn3

    def _apply_attn(self, S0, V0):
        ''' Multiplication with values '''
        nB, nH, nL, nL = S0.shape
        nD = V0.shape[-1]
        V1 = V0.reshape((nB*nH, nL, nD))

        S1 = S0.reshape((nB*nH, nL, nL))
        # S0: (nB, nH, nL, nL)
        # S1: (nB*nH, nL, nL)
        # V1:  (nB*nH, nL, nD)
        # S2: (nB*nH, nL, nD)
        # S3: (nB, nL, nH, nD)

        # (nB*nH, nL, nL) @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)

        S2 = self.ops.xp.matmul(S1, V1)

        S3 = S2.reshape((nB, nH, nL, nD)).transpose(0, 2, 1, 3)

        def backprop_attn4(dS3):
            # (nB, nL, nH, nD) --> (nB*nH, nL, nD)
            dS2 = dS3.transpose(0, 2, 1, 3).reshape((nB*nH, nL, nD))
            # (nB*nH, nL, nD) @ (nB*nH, nL, nD).T --> (nB*nH, nL, nL)
            dS1 = self.ops.xp.matmul(dS2, V1.transpose(0, 2, 1))
            # (nB*nH, nL, nL).T @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)
            dV1 = self.ops.xp.matmul(S1.transpose(0, 2, 1), dS2)
            dS0 = dS1.reshape((nB, nH, nL, nL))
            dV0 = dV1.reshape((nB, nH, nL, nD))
            return dS0, dV0

        return S3, backprop_attn4




class PytorchMultiHeadedAttention(nn.Module):
    def __init__(self, nM=300, nH=6, layer='Encoder'):
        super(PytorchMultiHeadedAttention, self).__init__()
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.layer = layer
        self.get_queries = nn.Linear(nM, nM)
        self.get_keys = nn.Linear(nM, nM)
        self.get_values = nn.Linear(nM, nM)
        self.get_output = nn.Linear(nM, nM)
        self._layers = [self.get_queries, self.get_keys, self.get_values, self.get_output]

    def forward(self, input, drop=0.0):
        # Queries come from input[0], keys and values from input[1]
        if len(input) == 3:
            x0, mask, sentX = input
            sentY = sentX
            y0 = x0
            self_attention = True
        else:
            self_attention = False
            x0, y0, mask, sentX, sentY = input
        mask = xp2torch(mask.astype(np.float32))
        ''' Shapes '''
        # x0: nB, nL, nM
        # q0: nB, nL, nM
        # k0: nB, nL, nM
        # v0: nB, nL, nM
        # q1: nB, nL, nH, nD
        # k1: nB, nL, nH, nD
        # v1: nB, nL, nH, nD
        # x1: nB, nL, nH, nD
        # x2: nB, nL, nM
        # x3: nB, nL, nM
        nB, nL, nD, nH = x0.shape[0], x0.shape[1], self.nD, self.nH
        q0 = self.get_queries(x0)
        q1 = q0.view(nB, -1, self.nH, self.nD)
        k0 = self.get_keys(y0)
        k1 = k0.view(nB, -1, self.nH, self.nD)
        v0 = self.get_values(y0)
        v1 = v0.reshape(nB, -1, self.nH, self.nD)
        x1 = self.attn(q1, k1, v1, mask=mask, sentX=sentX,
                                        sentY=sentY, self_attn=self_attention)
        x2 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2]*x1.shape[3])
        x3 = self.get_output(x2)
        return (x3, mask)

    def attn(self, Q, K, V, mask=None, sentX=None, sentY=None, self_attn=True):
        '''
        Compute attention on (query, key, value) triplets.
        The similarity of the (Q, K) pairs are used to
        compute an attention matrix, which is used to rescale
        V.
        '''
        # query shape: nB, nL, nH, nD

        S0 = self._scaled_dot_prod(Q, K)
        S1 = self._mask(S0, mask)
        S2 = self._softmax(S1)
        # if sentX is not None and sentY is not None:
        #     visualize_attention(sentX, sentY, S2[0], layer=self.layer,
        #     self_attn=self_attn)
        S3 = self._apply_attn(S2, V)
        return S3

    def _scaled_dot_prod(self, Q0, K0):
        # nB: #Sentences, nL: #Length, nH: #Heads, nD: #Dimensions
        nB, nL, nH, nD = Q0.shape
        # Shape of Q0: (nB, nL, nH, nD)
        # Shape of K0: (nB, nL, nH, nD)
        # --> (nB*nH, nL, nD)
        Q1 = Q0.permute(0, 2, 1, 3).contiguous().view(nB*nH, nL, nD)

        # --> (nB*nH, nD, nL)
        K1 = K0.permute(0, 2, 3, 1).contiguous().view(nB*nH, nD, nL)

        K2 = K1 / math.sqrt(self.nM)
        # (nB*nH, nL, nD) @ (nB*nH, nD, nL) --> (nB*nH, nL, nL)

        S = Q1 @ K2
        return S.view((nB, nH, nL, nL))

    def _mask(self, S0, mask):
        S1 = S0.permute(1, 0, 2, 3)
        S2 = S1 * mask - (1 - mask) * (1e9)
        S3 = S2.permute(1, 0, 2, 3)
        return S3

    def _softmax(self, S0):
        ''' A simple softmax to the scores '''
        # S0: nB, nH, nL, nL
        # S1: nB, nH, nL, nL
        softmax_layer = nn.Softmax(dim=-1)
        S1 = softmax_layer(S0)
        return S1

    def _apply_attn(self, S0, V0):
        ''' Multiplication with values '''
        nB, nH, nL, nL = S0.shape
        nD = V0.shape[-1]
        V1 = V0.view((nB*nH, nL, nD))

        S1 = S0.view((nB*nH, nL, nL))
        # S0: (nB, nH, nL, nL)
        # S1: (nB*nH, nL, nL)
        # V1:  (nB*nH, nL, nD)
        # S2: (nB*nH, nL, nD)
        # S3: (nB, nL, nH, nD)

        # (nB*nH, nL, nL) @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)

        S2 = S1 @ V1
        S3 = S2.view((nB, nH, nL, nD)).permute(0, 2, 1, 3)
        return S3
