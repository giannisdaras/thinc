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

    def begin_update(self, X, dim=3):
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

    def begin_update(self, batch, drop=0.1):
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

    def begin_update(self, batch, drop=0.1):
        batch, decoders_backprop = self.decoder_stack.begin_update(batch)
        print('Decoder stack computed output')
        return batch, decoders_backprop


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

        def finish_update(grad__BO):
            return attn_back(ffd_back(grad__BO))
        return batch, finish_update


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

        def finish_update(grad__BO):
            return slf_attn_back(other_attn_back(ffd_back(grad__BO)))
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
        value = value.reshape(nB, -1, self.heads, self.nK)
        X = self.attn(query, key, value, mask=mask)
        ''' sentences_in_batch x tokens_in_sentence x heads x head_vector '''
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
        X, out_backprop = self.linears[-1].begin_update(X)
        return X, None

    def attn(self, query, key, value, mask=None):
        ''' Compute attention on (query,key,value) triplet '''
        '''
        query shape:
        0: number of sentences
        1: number of tokens in the sentence
        2: number of heads
        3: vector dimension for each token of each head of each sentence
        '''
        scores = self.ops.xp.matmul(query.transpose(0, 2, 1, 3),
                                    key.transpose(0, 2, 3, 1) /
                                    math.sqrt(self.nI))
        scores = self.ops.softmax(scores)
        value = value.transpose(0, 2, 1, 3)
        real_scores = self.ops.xp.matmul(scores, value).transpose(0, 2, 1, 3)
        return real_scores
