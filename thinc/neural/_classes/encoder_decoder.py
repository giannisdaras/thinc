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
