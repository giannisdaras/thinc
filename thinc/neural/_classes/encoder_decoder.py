import math
import pdb
from .model import Model
from ...api import chain, clone, with_getitem, wrap
from .softmax import Softmax
from .layernorm import LayerNorm
from .resnet import Residual
from .affine import Affine


def with_reshape(layer):
    def with_reshape_forward(X, drop=0.):
        initial_shape = X.shape
        final_shape = list(initial_shape[:-1]) + [layer.nO]
        nB = X.shape[0]
        nT = X.shape[1]
        X2d = X.reshape(-1, X.shape[2])
        X2d = X2d.astype(layer.ops.xp.float32)
        Y2d, Y2d_backprop = layer.begin_update(X2d, drop=drop)
        Y = Y2d.reshape(final_shape)

        def with_reshape_backward(dY, sgd=None):
            dY = dY.reshape(nB*nT, -1).astype(layer.ops.xp.float32)
            return Y2d_backprop(dY, sgd=sgd).reshape(initial_shape)
        return Y, with_reshape_backward
    return wrap(with_reshape_forward, layer)


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
        self.enc = clone(EncoderLayer(self.nH, self.nM), self.nS)
        self.dec = clone(DecoderLayer(self.nH, self.nM), self.nS)
        self.proj = with_reshape(Softmax(nO=nTGT, nI=nM))
        self._layers = [self.enc, self.dec, self.proj]

    def begin_update(self, inputs, drop=0.0):
        '''
        A batch object flows through the network. It contains input, output and
        corresponding masks. Input changes while the object travels through
        the network. Output is the golden output.
        Input: nB x nL x nM
        '''
        (X0, Xmask), (Y0, Ymask) = inputs
        # b0: x0, y0
        # b1: x1, y1
        # b2: x2, y2
        (X1, _), get_dX0 = self.enc.begin_update((X0, Xmask), drop=drop)
        (_, (Y1, _)), get_dX1_dY1 = self.dec.begin_update(((X1, Xmask), (Y0, Ymask)), drop=drop)
        word_probs, get_dY1 = self.proj.begin_update(Y1, drop=drop)

        def finish_update(d_word_probs, sgd=None):
            dY1 = get_dY1(d_word_probs, sgd=sgd)
            zeros = Model.ops.xp.zeros(X1.shape, dtype=Model.ops.xp.float32)
            dX1, dY0 = get_dX1_dY1((zeros, dY1), sgd=sgd)
            dX0 = get_dX0(dX1, sgd=sgd)
            return (dX0, dY1)

        return (word_probs, Ymask), finish_update


def EncoderLayer(nH, nM):
    return chain(
        MultiHeadedAttention(nM, nH),
        with_getitem(0, Residual(with_reshape(Affine(nM, nM))))
    )


class DecoderLayer(Model):
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        ''' TODO: the following two layers should be probably residuals '''
        self.x_attn = MultiHeadedAttention(nM, nH)
        self.y_attn = MultiHeadedAttention(nM, nH)
        self.ffd = with_reshape(Affine(nM, nM))
        self._layers = [self.x_attn, self.y_attn, self.ffd]

    def begin_update(self, X_Y, drop=0.0):
        (X0, Xmask), (Y0, Ymask) = X_Y
        (Y1, _), get_dY00_dY01 = self.x_attn.begin_update((Y0, Y0, Ymask))
        (Y2, _), get_dY1_dX0 = self.y_attn.begin_update((Y1, X0, Xmask))
        Y3, get_dY2 = self.ffd.begin_update(Y2)

        def finish_update(dY3_dX0, sgd=None):
            dY3, dX = dY3_dX0
            dY2 = get_dY2(dY3, sgd=sgd)
            dY1, dX0 = get_dY1_dX0(dY2, sgd=sgd)
            dY00, dY01 = get_dY00_dY01(dY1, sgd=sgd)
            dY0 = dY00 + dY01
            dX += dX0
            return (dX, dY0,)

        return ((X0, Xmask), (Y3, Ymask)), finish_update


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
    def __init__(self, nM=300, nH=6):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM  # model size: the length of the embeddings
        self.nD = nM // nH
        self.linears = [with_reshape(Affine(nM, nM)) for i in range(4)]
        self._layers = list(self.linears)

    def begin_update(self, input, drop=0.0):
        if len(input) == 2:
            x0, mask = input
            y0 = x0
            self_attention = True
        else:
            self_attention = False
            x0, y0, mask = input
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
        q0, get_dx0 = self.linears[0].begin_update(x0)
        q1 = q0.reshape(nB, -1, self.nH, self.nD)
        k0, get_dy0_1 = self.linears[1].begin_update(y0)
        k1 = k0.reshape(nB, -1, self.nH, self.nD)
        v0, get_dy0_2 = self.linears[2].begin_update(y0)
        v1 = v0.reshape(nB, -1, self.nH, self.nD)

        x1, get_dq1_dk1_dv1 = self.attn(q1, k1, v1, mask=mask)

        x2 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2]*x1.shape[3])
        x3, get_dx2 = self.linears[-1].begin_update(x2)

        def finish_update(dx3, sgd=None):
            dx2 = get_dx2(dx3, sgd=sgd)
            dx1 = dx2.reshape(nB, nL, nH, nD)
            dq1, dk1, dv1 = get_dq1_dk1_dv1(dx1)
            dv0 = dv1.reshape(nB, nL, nH, nD)
            dk0 = dk1.reshape(nB, nL, nH, nD)
            dq0 = dq1.reshape(nB, nL, nH, nD)
            dy0 = get_dy0_2(dv0, sgd=sgd)
            dy0 += get_dy0_1(dk0, sgd=sgd)
            dx0 = get_dx0(dq0, sgd=sgd)
            if self_attention:
                return dx0 + dy0
            else:
                return (dx0, dy0)
        return (x3, mask), finish_update

    def attn(self, Q, K, V, mask=None):
        ''' Compute attention on (query, key, value) triplet '''
        # query shape: nB, nL, nH, nD

        S0, get_dQ_dK = self._attn1(Q, K)
        S1, get_dS0 = self._attn2(S0, mask)
        S2, get_dS1 = self._attn3(S1)
        S3, get_dS2_dV = self._attn4(S2, V)

        def backprop_attn(dS3):
            ''' Attention three inputs, one output '''
            dS2, dV = get_dS2_dV(dS3)
            dS1 = get_dS1(dS2)
            dS0 = get_dS0(dS1)
            dQ, dK = get_dQ_dK(dS0)
            return dQ, dK, dV
        return S3, backprop_attn

    def _attn1(self, Q0, K0):
        # nB: #Sentences, nL: #Length, nH: #Heads, nD: #Dimensions
        nB, nL, nH, nD = Q0.shape
        # Shape of Q0: (nB, nL, nH, nD)
        # Shape of K0: (nB, nL, nH, nD)
        # --> (nB*nH, nL, nD)

        Q1 = Q0.transpose(0, 2, 1, 3).reshape(nB*nH, nL, nD)

        # --> (nB*nH, nD, nL)
        K1 = K0.transpose(0, 2, 3, 1).reshape(nB*nH, nD, nL)

        K2 = K1 / math.sqrt(self.nM)
        # (nB*nH, nL, nD) @ (nB*nH, nD, nL) --> (nB*nH, nL, nL)

        S = self.ops.xp.matmul(Q1, K2)

        def backprop_attn1(dS):
            # (nB*nH, nL, nL) @ (nB*nH, nD, nL).T --> (nB*nH, nL, nD)
            dS = dS.reshape(nB*nH, nL, nL)
            dQ1 = self.ops.xp.matmul(dS, K2.transpose(0, 2, 1))
            # (nB*nH, nL, nD).T @ (nB*nH, nL, nL) --> (nB*nH, nD, nL)
            dK2 = self.ops.xp.matmul(Q1.transpose(0, 2, 1), dS)
            dK1 = dK2 / math.sqrt(self.nM)
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 2, 3, 1)
            dQ0 = dQ1.reshape((nB, nH, nL, nD)).transpose(0, 2, 1, 3)
            return dQ0, dK0
        return S.reshape((nB, nH, nL, nL)), backprop_attn1

    def _attn2(self, S0, mask):
        S1 = S0.transpose(1, 0, 2, 3)
        S2 = S1 * mask - (1 - mask) * (1e9)
        S3 = S2.transpose(1, 0, 2, 3)

        def backprop_attn2(dS3):
            dS2 = dS3.transpose(1, 0, 2, 3)
            dS1 = dS2 * mask
            dS0 = dS1.transpose(1, 0, 2, 3)
            return dS0

        return S3, backprop_attn2

    def _attn3(self, S0):
        ''' A simple softmax to the scores '''
        # S0: nB, nH, nL, nL
        # S1: nB, nH, nL, nL
        S1 = self.ops.softmax(S0)

        def backprop_attn3(dS1):
            dS0 = self.ops.xp.matmul(dS1, self.ops.xp.matmul(S0, (1 - S0)))
            return dS0
        return S1, backprop_attn3

    def _attn4(self, S0, V0):
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
