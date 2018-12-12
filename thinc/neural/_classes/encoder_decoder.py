from .model import Model
from ...api import chain, clone
from .softmax import Softmax
from .layernorm import LayerNorm
from .resnet import Residual
from ...linear.linear import LinearModel


class EncoderDecoder:
    def __init__(self, **kwargs):
        self.stack = kwargs.get('stack', 6)
        self.heads = kwargs.get('heads', 6)
        self.model_size = kwargs.get('model_size', 300)
        self.tgt_vocab_size = kwargs.get('tgt_vocab_size', 10000)
        self.enc = LayerNorm(EncoderLayer(self.heads, seld.model_size, self.stack))
        self.dec = DecoderLayer(self.heads, self.model_size) ** self.stack
        self.output_layer = LinearModel(self.model_size, self.tgt_vocab_size)

    def begin_update(self, batch):
        X = batch.X
        y = batch.y
        X_mask = batch.X_mask
        y_mask = batch.y_mask
        batch_size = batch.batch_size
        enc_out, enc_backprop = self.enc.begin_update(X, X_mask)
        dec_out, dec_backprop = self.dec.begin_update(X, y, X_mask, y_mask)
        output, output_backprop = self.output_layer.begin_update()
        # possibly wrong?
        return output, [output_backprop, dec_backprop, enc_backprop]


class Encoder:
    def __init__(self, heads, model_size, stack):
        self.heads = heads
        self.model_size = model_size
        self.stack = stack
        self.encoder_stack = [EncoderLayer(heads, model_size) for i in range(stack)]

    def begin_update(self, X, X_mask):
        backprops = []
        for layer in self.encoder_stack:
            X, layer_backprop = layer(X, X_mask)
            backprops.append(layer_backprop)
        return X, backprops


class Decoder:
    def __init__(self, heads, model_size, stack):
        self.heads = heads
        self.model_size = model_size
        self.stack = stack
        self.decoder_stack = [DecoderLayer(heads, model_size) for i in range(stack)]

    def begin_update(self, X, y, X_mask, y_mask):
        backprops = []
        for layer in self.decoder_stack:
            X, layer_backprop = layer(X, y, X_mask, y_mask)
            backprops.append(layer_backprop)
        return X, backprops


class EncoderLayer:
    def __init__(heads, model_size):
        self.heads = heads
        self.model_size
        self.attention = MultiHeadedAttention(model_size, heads)
        self.ffd = Linear(model_size, model_size)
        self.residuals = [Residual(self.attention), Residual(self.ffd)]

    def begin_update(X, X_mask):
        X, attn_back = self.residuals[0].begin_update(X,
                                lambda x: self.attention(x, x, X_mask))
        X, ffd_back = self.residuals[1].begin_update(X, self.ffd)
        # possibly wrong?
        return X, [ffd_back, attn_back]


class DecoderLayer:
    def __init__(heads, model_size):
        self.heads = heads
        self.model_size
        self.slf_attention = MultiHeadedAttention(model_size, heads)
        self.other_attention = MultiHeadedAttention(model_size, heads)
        self.ffd = Linear(model_size, model_size)
        self.residuals = [Residual(self.slf_attention),
                          Residual(self.other_attention),
                          Residual(self.ffd)
                          ]

    def begin_update(X, y, X_mask, y_mask):
        y, slf_attn_back = self.residuals[0].begin_update(y, lambda x: self.slf_attention(y, y, y_mask))
        y, other_attn_back = self.residuals[1].begin_update(y, lambda x: self.other_attention(y, x, x_mask))
        y, ffd_back = self.ffd.begin_update(y)
        return y, [ffd_back, other_attn_back, slf_attn_back]

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
    def __init__(self, nI, heads=6):
        self.heads = heads
        self.nI = nI  # model size: the length of the embeddings
        self.nK = nI // heads
        self.linears = [LinearModel(nI, nI) for i in range(4)]

    def begin_update(X, y, mask):
        nB = X.shape[0]
        query, query_backprop = self.linears[0].begin_update(X)
        query = query.reshape(nB, -1, self.heads, self.nK)
        key, key_backprop = self.linears[1].begin_update(y)
        key = key.reshape(nB, -1, self.heads, self.nK)
        value, value_backprop = self.linears[2].begin_update(y)
        value.reshape(nB, -1, self.heads, self.nK)
        X = attn(query, key, value, mask=mask)
        X = X.reshape(1, 2).reshape(nB, -1, self.heads * self.nK)
        X, out_backprop = self.linears[-1].begin_update(X)
        return X, [out_backprop, value_backprop, key_backprop, query_backprop]

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
