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
