import math
import pdb
from .model import Model
from ...api import chain, clone
from .softmax import Softmax
from .layernorm import LayerNorm
from .resnet import Residual
from .affine import Affine


class SeqLinear(Model):
    def __init__(self, nI=300, nO=300):
        Model.__init__(self)
        self.nI = nI
        self.nO = nO
        self.linear = Affine(nI=nI, nO=nO)

    def begin_update(self, X, drop=0.0, dim=3):
        initial_shape = X.shape
        final_shape = list(initial_shape[:-1]) + [self.nO]
        nB = X.shape[0]
        nT = X.shape[1]
        X2d = X.reshape(-1, X.shape[2])
        Y2d, Y2d_backprop = self.linear.begin_update(X2d)
        Y = Y2d.reshape(final_shape)

        def finish_update(grad__BO):
            grad__BO = grad__BO.reshape(nB*nT, -1)
            return Y2d_backprop(grad__BO).reshape(initial_shape)
        return Y, finish_update


class SeqSoftmax(Model):
    def __init__(self, nI=300, nO=300):
        Model.__init__(self)
        self.nI = nI
        self.nO = nO
        self.softmax = Softmax(nI=nI, nO=nO)

    def begin_update(self, X, dim=3):
        initial_shape = X.shape
        final_shape = list(initial_shape[:-1]) + [self.nO]
        nB = X.shape[0]
        nT = X.shape[1]
        X2d = X.reshape(-1, X.shape[2])
        Y2d, Y2d_backprop = self.softmax.begin_update(X2d)
        Y = Y2d.reshape(final_shape)

        def finish_update(grad__BO):
            grad__BO = grad__BO.reshape(nB*nT, Y.shape[-1])
            return Y2d_backprop(grad__BO).reshape(initial_shape)
        return Y, finish_update


class EncoderDecoder(Model):
    def __init__(self, stack=6, heads=6, model_size=300, tgt_vocab_size=10000):
        '''
        EncoderDecoder consists of an encoder stack, a decoder stack and an
        output layer which is a linear + softmax.
        Parameters explanation:
            stack: the number of encoders/decoders in the stack
            heads: the number of heads in the multiheaded attention
            model_size: the token's embedding size
            tgt_vocab_size: the number of unique words in output vocabulary
        '''
        Model.__init__(self)
        self.stack = stack
        self.heads = heads
        self.model_size = model_size
        self.tgt_vocab_size = tgt_vocab_size
        self.enc = Encoder(self.heads, self.model_size, self.stack)
        self.dec = Decoder(self.heads, self.model_size, self.stack)
        self.output_layer = SeqSoftmax(model_size, tgt_vocab_size)

    def begin_update(self, batch, drop=0.0):
        '''
        A batch object flows through the network. It contains input, output and
        corresponding masks. Input changes while the object travels through
        the network. Output is the golden output.

        Input: sentences_in_batch x tokens_per_sentence x model_size
        '''
        enc_out, enc_backprop = self.enc.begin_update(batch)
        dec_out, dec_backprop = self.dec.begin_update(batch)
        y = dec_out.y
        output, output_backprop = self.output_layer.begin_update(y)

        def finish_update(grad__BO):
            return enc_backprop(dec_backprop(output_backprop(grad__BO)))
        return output, finish_update


class Encoder(Model):
    def __init__(self, heads, model_size, stack):
        Model.__init__(self)
        self.heads = heads
        self.model_size = model_size
        self.stack = stack
        self.encoder_stack = EncoderLayer(heads, model_size)
        for i in range(self.stack - 1):
            self.encoder_stack = chain(self.encoder_stack,
                                       EncoderLayer(heads, model_size))

    def begin_update(self, batch, drop=0.0):
        batch, encoders_backprop = self.encoder_stack.begin_update(batch)
        return batch, encoders_backprop


class Decoder(Model):
    def __init__(self, heads, model_size, stack):
        Model.__init__(self)
        self.heads = heads
        self.model_size = model_size
        self.stack = stack
        self.decoder_stack = DecoderLayer(heads, model_size)
        for i in range(self.stack - 1):
            self.decoder_stack = chain(self.decoder_stack,
                                       DecoderLayer(heads, model_size))

    def begin_update(self, batch, drop=0.0):
        batch, decoders_backprop = self.decoder_stack.begin_update(batch)
        print('Decoder stack computed output')
        return batch, decoders_backprop


class EncoderLayer(Model):
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        ''' TODO: this layer should be probably made residual '''
        self.x_attn = MultiHeadedAttention(nM, nH)
        self.ffd = Residual(SeqLinear(nM, nM))

    def begin_update(self, batch, drop=0.0):
        x0 = batch.X
        X_mask = batch.X_mask
        x1, get_dx00_dx01 = self.attention.begin_update((x0, x0, X_mask))
        x2, get_dx1 = self.ffd.begin_update(x1)
        batch.X = x2

        def finish_update(dx2):
            dx1 = get_dx1(dx2)
            dx00, dx01 = get_dx00_dx01(dx1)
            dx = dx00 + dx01
            return dx
        return batch, finish_update


class DecoderLayer(Model):
    def __init__(self, nH, nM):
        Model.__init__(self)
        self.nH = nH
        self.nM = nM
        ''' TODO: the following two layers should be probably residuals '''
        self.x_attn = MultiHeadedAttention(nM, nH)
        self.y_attn = MultiHeadedAttention(nM, nH)
        self.ffd = SeqLinear(nM, nM)
        self.residuals = [self.x_attn, self.y_attn, Residual(self.ffd)]

    def begin_update(self, batch, drop=0.0):
        x0 = batch.X
        y0 = batch.y
        X_mask = batch.X_mask
        y_mask = batch.y_mask
        y1, get_dy00_dy01 = self.residuals[0].begin_update((y0, y0, y_mask))
        y2, get_dy1_dx0 = self.residuals[1].begin_update((y1, x0, X_mask))
        y3, get_dy2 = self.ffd.begin_update(y2)
        batch.y = y3

        def finish_update(dy3):
            dy2 = get_dy2(dy3)
            dy1, dx0 = get_dy1_dx0(dy2)
            dy00, dy01 = get_dy00_dy01(dy1)
            dy0 = dy00 + dy01
            return (dx0, dy0)

        return batch, finish_update


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
        self.linears = [SeqLinear(nM, nM) for i in range(4)]

    def begin_update(self, input, drop=0.0):
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
        q1 = q0.reshape(nB, -1, self.heads, self.nK)
        k0, get_dy0_1 = self.linears[1].begin_update(y0)
        k1 = k0.reshape(nB, -1, self.heads, self.nK)
        v0, get_dy0_2 = self.linears[2].begin_update(y0)
        v1 = v0.reshape(nB, -1, self.heads, self.nK)
        x1, get_dq1_dk1_dv1 = self.attn(q1, k1, v1, mask=mask)
        x2 = x1.reshape(x1.shape[0], x1.shape[1], x1.shape[2]*x1.shape[3])
        x3, get_dx2 = self.linears[-1].begin_update(x2)

        def finish_update(dx3):
            dx2 = get_dx2(dx3)
            dx1 = dx2.reshape(nB, nL, nH, nD)
            dq1, dk1, dv1 = get_dq1_dk1_dv1(dx1)
            dv0 = dv1.reshape(nB, nL, nH, nD)
            dk0 = dk1.reshape(nB, nL, nH, nD)
            dq0 = dq1.reshape(nB, nL, nH, nD)
            dy0 = get_dy0_2(dv0)
            dy0 += get_dy0_1(dk0)
            dx0 = get_dx0(dq0)
            return (dx0, dy0)

        return x3, finish_update

    def attn(self, Q, K, V, mask=None):
        ''' Compute attention on (query, key, value) triplet '''
        # query shape: nB, nL, nH, nD
        S0, get_dQ_dK = self._attn1(Q, K)
        S1, get_dS0 = self._attn2(S0)
        S2, get_dS1_dV = self._attn3(S1, V)

        def backprop_attn(dS2):
            ''' Attention three inputs, one output '''
            dS1, dV = get_dS1_dV(dS2)
            dS0 = get_dS0(dS1)
            dQ, dK = get_dQ_dK(dS0)
            return dQ, dK, dV
        return S2, backprop_attn

    def _attn1(self, Q0, K0):
        # nB: #Sentences, nL: #Length, nH: #Heads, nD: #Dimensions
        nB, nL, nH, nD = Q0.shape
        # Shape of Q0: (nB, nL, nH, nD)
        # Shape of K0: (nB, nL, nH, nD)

        # --> (nB*nH, nL, nD)
        Q1 = Q0.transpose(0, 2, 1, 3).reshape(nB*nH, nL, nD)
        # --> (nB*nH, nD, nL)
        K1 = K0.transpose(0, 2, 3, 1).reshape(nB*nH, nD, nL)
        K2 = K1 / math.sqrt(self.nI)
        # (nB*nH, nL, nD) @ (nB*nH, nD, nL) --> (nB*nH, nL, nL)
        S = self.ops.xp.matmul(Q1, K2)

        def backprop_attn1(dS):
            # (nB*nH, nL, nL) @ (nB*nH, nD, nL).T --> (nB*nH, nL, nD)
            # To test this, set some values in dS to nan, and check they
            # propagate how we expect.
            # Also can compare against an autograd solution.
            dQ1 = self.ops.xp.matmul(dS, K2.transpose(0, 2, 1))
            # (nB*nH, nL, nD).T @ (nB*nH, nL, nL) --> (nB*nH, nD, nL)
            dK2 = self.ops.xp.matmul(Q1.transpose((0, 2, 1), dS))
            dK1 = dK2 / math.sqrt(self.nI)
            dK0 = dK1.reshape((nB, nH, nD, nL)).transpose(0, 2, 3, 1)
            dQ0 = dQ1.reshape((nB, nH, nL, nD)).transpose(0, 2, 1, 3)
            return dQ0, dK0
        return S.reshape((nB, nH, nL, nL)), backprop_attn1

    def _attn2(self, S0):
        ''' A simple softmax to the scores '''
        # S0: nB, nH, nL, nL
        # S1: nB, nH, nL, nL
        S1 = self.ops.softmax(S0)

        def backprop_attn2(dS1):
            dS0 = self.ops.xp.matmul(dS1, self.ops.xp.matmul(S0, (1 - S0)))
            return dS0
        return S1, backprop_attn2

    def _attn3(self, S0, V0):
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

        def backprop_attn3(dS3):
            # (nB, nL, nH, nD) --> (nB*nH, nL, nD)
            dS2 = dS3.tranpose(0, 2, 1, 3).reshape((nB*nH, nH, nD))
            # (nB*nH, nL, nD) @ (nB*nH, nL, nD).T --> (nB*nH, nL, nL)
            dS1 = self.ops.xp.matmul(dS2, V1.transpose(0, 2, 1))
            # (nB*nH, nL, nL).T @ (nB*nH, nL, nD) --> (nB*nH, nL, nD)
            dV1 = self.ops.xp.matmul(S1.transpose(0, 2, 1), dS2)
            dS0 = dS1.reshape((nB, nH, nL, nL))
            dV0 = dV1.reshape((nB, nL, nL, nD))
            return dS0, dV0

        return S3, backprop_attn3
