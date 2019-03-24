''' A driver file for attention is all you need paper demonstration '''
from __future__ import unicode_literals
import plac
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, ORTH
from thinc.extra.datasets import get_iwslt
from thinc.loss import categorical_crossentropy
from thinc.neural.util import get_array_module
from spacy._ml import link_vectors_to_models
from thinc.neural.util import to_categorical
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.embed import Embed
from thinc.api import wrap, chain, with_flatten, layerize
from thinc.misc import Residual
from thinc.v2v import Model
from timeit import default_timer as timer

from spacy.lang.en import English
from spacy.lang.de import German
import pickle
import sys
MODEL_SIZE = 300
MAX_LENGTH = 50
VOCAB_SIZE = 5000


class Batch:
    ''' Objects of this class share pairs X,Y along with their
        padding/future-words-hiding masks '''
    def __init__(self, pair, pad_masks, lengths):
        # X, y: nB x nL x nM
        # X_pad_mask, y_pad_mask: nB x nL
        # X_mask, y_mask: nB x nL x nL
        X, y = pair
        self.X_pad_mask, self.y_pad_mask = pad_masks
        nX, nY = lengths
        ''' TODO: this is super clumpsy and we need to fix this, but at least
        is happening only once ber batch, so it's acceptable for now '''
        self.y = y
        self.nB = X.shape[0]
        self.nL = X.shape[1]
        self.X = X
        self.X_mask = Model.ops.xp.repeat(self.X_pad_mask[:, :, Model.ops.xp.newaxis], self.nL, axis=2)
        ''' Explanation:
        y_mask has different dimensions than x_pad mask, because the mask
        itself is responsible for the different "steps" in the decoder
        architecture. That actually means, that y_mask should not only mask the
        pad tokens, but take care of avoiding attention to future positions.
        y_mask has shape: nB x nL x nL, to hold the different masks for the
        different tokens. To get to that 3d array from y_pad_mask and
        subsequent mask (which are 2d arrays), we insert a fake dimension
        for successfull broadcasting.
        '''
        # (nB, 1, nL) & (1, nL, nL) --> (nB, nL, nL)
        y_pad_mask_expand = Model.ops.xp.expand_dims(self.y_pad_mask, -2)
        subsequent_mask = self.subsequent_mask(self.nL)
        assert y_pad_mask_expand.shape == (self.nB, 1, self.nL)
        assert subsequent_mask.shape == (1, self.nL, self.nL)
        self.y_mask = y_pad_mask_expand & subsequent_mask

    def subsequent_mask(self, nL):
        return (Model.ops.xp.triu(Model.ops.xp.ones([1, nL, nL]), k=1) == 0)


def spacy_tokenize(X_tokenizer, Y_tokenizer, X, Y, max_length=50):
    X_out = []
    Y_out = []
    for x, y in zip(X, Y):
        xdoc = X_tokenizer('<bos> ' + x.strip() + ' <eos>')
        ydoc = Y_tokenizer('<bos> ' + y.strip() + ' <eos>')
        if len(xdoc) < MAX_LENGTH and (len(ydoc) + 2) < MAX_LENGTH:
            X_out.append(xdoc)
            Y_out.append(ydoc)
    return X_out, Y_out



def PositionEncode(max_length, model_size):
    positions = Model.ops.position_encode(max_length, model_size)
    def position_encode_forward(Xs, drop=0.):
        output = []
        for x in Xs:
            output.append(positions[:x.shape[0]])
        return output, None
    return layerize(position_encode_forward)


@layerize
def docs2ids(docs, drop=0.):
    """Extract ids from a batch of (docx, docy) tuples."""
    ops = Model.ops
    ids = []
    for doc in docs:
        if "ids" not in doc.user_data:
            doc.user_data["ids"] = ops.asarray(doc.to_array(ID), dtype='int32')
        ids.append(doc.user_data["ids"])
    return ids, None


def apply_layers(*layers):
    """Take a sequence of input layers, and expect input tuples. Apply
    layers[0] to inputs[1], layers[1] to inputs[1], etc.
    """
    def apply_layers_forward(inputs, drop=0.):
        assert len(inputs) == len(layers), (len(inputs), len(layers))
        outputs = []
        callbacks = []
        for layer, input_ in zip(layers, inputs):
            output, callback = layer.begin_update(input_, drop=drop)
            outputs.append(output)
            callbacks.append(callback)

        def apply_layers_backward(d_outputs, sgd=None):
            d_inputs = []
            for callback, d_output in zip(callbacks, d_outputs):
                if callback is None:
                    d_inputs.append(None)
                else:
                    d_inputs.append(callback(d_output, sgd=sgd))
            return d_inputs
        return tuple(outputs), apply_layers_backward
    return wrap(apply_layers_forward, *layers)


def set_numeric_ids(vocab, docs, vocab_size=0, force_include=("<eos>", "<bos>")):
    """Count word frequencies and use them to set the lex.rank attribute."""
    freqs = Counter()
    for doc in docs:
        for token in doc:
            freqs[token.orth] += 1
    rank = 1
    for lex in vocab:
        lex.rank = 0
    for word in force_include:
        lex = vocab[word]
        lex.rank = rank
        rank += 1
    for orth, count in enumerate(freqs.most_common()):
        lex = vocab[orth]
        if lex.text not in force_include:
            rank += 1
            lex.rank = rank
        if vocab_size != 0 and rank >= vocab_size:
            break


def resize_vectors(vectors):
    xp = get_array_module(vectors.data)
    shape = (int(vectors.shape[0]*1.1), vectors.shape[1])
    if not hasattr(xp, 'resize'):
        vectors.data = vectors.data.get()
        vectors.resize(shape)
        vectors.data = xp.array(vectors.data)
    else:
        vectors.resize(shape)


def create_batch():
    def create_batch_forward(Xs_Ys, drop=0.):
        Xs, Ys = Xs_Ys
        nX = model.ops.asarray([x.shape[0] for x in Xs], dtype='i')
        nY = model.ops.asarray([y.shape[0] for y in Ys], dtype='i')

        nL = max(nX.max(), nY.max())
        Xs, _, unpad_dXs = model.ops.square_sequences(Xs, pad_to=nL)
        Ys, _, unpad_dYs = model.ops.square_sequences(Ys, pad_to=nL)
        Xs = Xs.transpose((1, 0, 2))
        Ys = Ys.transpose((1, 0, 2))

        X_mask = _create_mask(model.ops, nX, nL)
        Y_mask = _create_mask(model.ops, nY, nL)

        def create_batch_backward(dXs_dYs, sgd=None):
            dXs, dYs = dXs_dYs
            dXs = unpad_dXs(dXs.transpose((1, 0, 2)))
            dYs = unpad_dYs(dYs.transpose((1, 0, 2)))
            return dXs, dYs

        batch = Batch((Xs, Ys), (X_mask, Y_mask), (nX, nY))
        return batch, create_batch_backward
    model = layerize(create_batch_forward)
    return model


def _create_mask(ops, lengths, max_sent):
    batch_size = len(lengths)
    mask = ops.xp.ones([batch_size, max_sent], dtype=ops.xp.int)
    for i in range(len(lengths)):
        pad_size = max_sent - lengths[i]
        # !his if is a bit ugly, but slicing gets really
        # weird if you end up with a zero here.
        if pad_size > 0:
            mask[i, pad_size:] = 0
    return mask


def get_loss(ops, Yh, Y_docs):
    Y_ids = docs2ids(Y_docs)
    nC = Yh.shape[-1]
    Y = [to_categorical(y, nb_classes=nC) for y in Y_ids]
    nL = max(Yh.shape[1], max(y.shape[0] for y in Y))
    Y, _, _ = ops.square_sequences(Y, pad_to=nL)
    return Yh-Y.transpose((1, 0, 2))

@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option", "nH", int),
    dropout=("model dropout", "option"),
    nS=('Number of encoders/decoder in the enc/dec stack.', "option", "nS", int),
    nB=('Batch size for the training', "option", "nB", int),
    nE=('Number of epochs for the training', "option", "nE", int),
    use_gpu=("Which GPU to use. -1 for CPU", "option", "g", int),
    lim=("Number of sentences to load from dataset", "option", "l", int)
)
def main(nH=6, dropout=0.1, nS=6, nB=15, nE=20, use_gpu=-1, lim=2000):
    if use_gpu != -1:
        # TODO: Make specific to different devices, e.g. 1 vs 0
        spacy.require_gpu()
    train, dev, test = get_iwslt()
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    test_X, test_Y = zip(*test)
    ''' Read dataset '''
    nlp_en = spacy.load('en_core_web_sm')
    nlp_de = spacy.load('de_core_news_sm')
    print('Models loaded')
    for control_token in ("<eos>", "<bos>", "<pad>"):
        nlp_en.tokenizer.add_special_case(control_token, [{ORTH: control_token}])
        nlp_de.tokenizer.add_special_case(control_token, [{ORTH: control_token}])
    train_X, train_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                      train_X[-lim:], train_Y[-lim:], MAX_LENGTH)
    dev_X, dev_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                  dev_X[-lim:], dev_Y[-lim:], MAX_LENGTH)
    test_X, test_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                    test_X[-lim:], test_Y[-lim:], MAX_LENGTH)
    set_numeric_ids(nlp_en.vocab, train_X, vocab_size=VOCAB_SIZE,
        force_include=("<eos>", "<bos>"))
    set_numeric_ids(nlp_de.vocab, train_Y)
    nTGT = VOCAB_SIZE

    with Model.define_operators({">>": chain}):
        position_encode = PositionEncode(MAX_LENGTH, MODEL_SIZE)
        model = (
            apply_layers(docs2ids, docs2ids)
            >> apply_layers(
                with_flatten(Embed(MODEL_SIZE, nM=MODEL_SIZE, nV=nTGT)),
                with_flatten(Embed(MODEL_SIZE, nM=MODEL_SIZE, nV=nTGT)))
            >> apply_layers(Residual(position_encode), Residual(position_encode))
            >> create_batch()
            >> EncoderDecoder(nTGT=nTGT)
        )

    losses = [0.]
    def track_progress():
        print(len(losses), losses[-1])
        losses.append(0.)


    with model.begin_training(batch_size=nB, nb_epoch=nE) as (trainer, optimizer):
        trainer.dropout = dropout
        trainer.dropout_decay = 1e-4
        trainer.each_epoch.append(track_progress)
        for X, Y in trainer.iterate(train_X, train_Y):
            Yh, backprop = model.begin_update((X, Y), drop=0.2)
            dYh = get_loss(model.ops, Yh, Y)
            backprop(dYh, sgd=optimizer)
            losses[-1] += (dYh**2).sum()


if __name__ == '__main__':
    plac.call(main)
