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
        self.W = Model.ops.xp.random.rand(nO, nI)

    def begin_update(self, X, dim=3):
        X2d = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        Y2d = Model.ops.gemm(X2d, self.W, trans2=True)

        def backward(dY):
            ''' todo complete this '''
            return None
        if dim == 3:
            return Y2d.reshape(X.shape[0], X.shape[1], X.shape[2]), backward
        else:
            return Y2d, backward


class SeqSoftmax(Model):
    def __init__(self, nI=300, nO=300):
        Model.__init__(self)
        self.linear = SeqLinear(nI, nO)

    def begin_update(self, X):
        out, linear_backprop = self.linear.begin_update(X, dim=2)
        return self.ops.softmax(out), None


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

    def begin_update(self, batch, drop=0.1):
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
        return output, None


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

    def begin_update(self, batch, drop=0.1):
        batch, encoders_backprop = self.encoder_stack.begin_update(batch)
        print('Encoder stack computed output')
        return batch, None


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

    def begin_update(self, batch, drop=0.1):
        batch, decoder_backprop = self.decoder_stack.begin_update(batch)
        print('Decoder stack computed output')
        return batch, None


class EncoderLayer(Model):
    def __init__(self, heads, model_size):
        Model.__init__(self)
        self.heads = heads
        self.model_size = model_size
        self.attention = MultiHeadedAttention(model_size, heads)
        self.ffd = SeqLinear(model_size, model_size)

    def begin_update(self, batch, drop=0.1):
        X = batch.X
        X_mask = batch.X_mask
        X, attn_back = self.attention.begin_update(X, X, X_mask)
        X, ffd_back = self.ffd.begin_update(X)
        batch.X = X
        return batch, None


class DecoderLayer(Model):
    def __init__(self, heads, model_size):
        Model.__init__(self)
        self.heads = heads
        self.model_size = model_size
        self.slf_attention = MultiHeadedAttention(model_size, heads)
        self.other_attention = MultiHeadedAttention(model_size, heads)
        self.ffd = SeqLinear(model_size, model_size)
        self.residuals = [self.slf_attention,
                          self.other_attention,
                          self.ffd
                          ]

    def begin_update(self, batch, drop=0.1):
        X = batch.X
        y = batch.y
        X_mask = batch.X_mask
        y_mask = batch.y_mask
        y, slf_attn_back = self.residuals[0].begin_update(y, y, y_mask)
        y, other_attn_back = self.residuals[1].begin_update(y, X, X_mask)
        y, ffd_back = self.ffd.begin_update(y)
        batch.y = y
        return batch, None


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
    def __init__(self, nI=300, heads=6):
        Model.__init__(self)
        self.heads = heads
        self.nI = nI  # model size: the length of the embeddings
        self.nK = nI // heads
        self.linears = [SeqLinear(nI, nI) for i in range(4)]

    def begin_update(self, X, y, mask, drop=0.1):
        nB = X.shape[0]
        query, query_backprop = self.linears[0].begin_update(X)
        query = query.reshape(nB, -1, self.heads, self.nK)
        key, key_backprop = self.linears[1].begin_update(y)
        key = key.reshape(nB, -1, self.heads, self.nK)
        value, value_backprop = self.linears[2].begin_update(y)
        value.reshape(nB, -1, self.heads, self.nK)
        X = self.attn(query, key, value, mask=mask)
        X = X.reshape(1, 2).reshape(nB, -1, self.heads * self.nK)
        X, out_backprop = self.linears[-1].begin_update(X)
        return X, None

    def attn(self, query, key, value, mask=None):
        ''' Compute attention based on query, key, value
        Expected data size: nB x -1 x nT x nK
        '''
        nB = query.shape[0]
        # number of tokens in each sentence
        nT = query.shape[2]
        # key length
        nK = query.shape[3]
        scores = self.ops.xp.matmul(query, key.transpose(0, 1, 3, 2)) \
            / math.sqrt(self.nI)
        # penalize masked tokens
        scores[self.ops.xp.where(mask == 0)] = 1e-9
        # TODO: fix this!
        # this could be done much faster if softmax was supported for >= 3d.
        for batch in range(nB):
            scores[batch, :, :, :] = self.ops.softmax(scores[batch, :, :])
        ''' Now the dimensions of scores are:
        nB x nT x nT
        We multiply by values which is nB x nT x nK, so the result is:
        nB x nT x nK
        '''
        real_scores = self.ops.xp.matmul(scores, value)
        return real_scores
