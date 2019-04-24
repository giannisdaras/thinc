''' A driver file for attention is all you need paper demonstration '''
from __future__ import unicode_literals
from collections import defaultdict
import random
import numpy
import plac
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, ORTH, SHAPE, PREFIX, SUFFIX
from thinc.extra.datasets import get_iwslt
from thinc.loss import categorical_crossentropy
from thinc.neural.util import get_array_module
from spacy._ml import link_vectors_to_models
from thinc.neural.util import to_categorical, minibatch
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.embed import Embed
from thinc.misc import FeatureExtracter
from thinc.api import wrap, chain, with_flatten, layerize
from thinc.misc import Residual
from thinc.v2v import Model
from timeit import default_timer as timer
import numpy.random
import random
from thinc.extra.wrappers import PyTorchWrapper
from spacy.lang.en import English
from spacy.lang.de import German
import pickle
import sys

random.seed(0)
numpy.random.seed(0)


class Batch:
    ''' Objects of this class share pairs X,Y along with their
        padding/future-words-hiding masks '''
    def __init__(self, pair, lengths):
        # X, y: nB x nL x nM
        # X_pad_mask, y_pad_mask: nB x nL
        # X_mask, y_mask: nB x nL x nL
        X, y = pair
        nX, nY = lengths
        self.X = X
        self.y = y
        self.nB = X.shape[0]
        self.nL = X.shape[1]
        self.X_mask = Model.ops.allocate((self.nB, self.nL, self.nL))
        self.y_mask = Model.ops.allocate((self.nB, self.nL, self.nL))
        for i, length in enumerate(nX):
            self.X_mask[i, :, :length] = 1.0
        for i, length in enumerate(nY):
            for j in range(length):
                self.y_mask[i, j, :j+1] = 1.0
            self.y_mask[i, length:, :length] = 1.0


def spacy_tokenize(X_tokenizer, Y_tokenizer, X, Y, mL=50):
    X_out = []
    Y_out = []
    for x, y in zip(X, Y):
        xdoc = X_tokenizer(x.strip())
        ydoc = Y_tokenizer('<bos> ' + y.strip() + ' <eos>')
        if len(xdoc) < mL and (len(ydoc) + 2) < mL:
            X_out.append(xdoc)
            Y_out.append(ydoc)
    return X_out, Y_out


def PositionEncode(mL, nM):
    positions = Model.ops.position_encode(mL, nM)
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
        if not (ids[-1] != 0).all():
            raise ValueError(ids[-1])
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


def set_numeric_ids(vocab, docs, nTGT=0, force_include=("<oov>", "<eos>", "<bos>")):
    """Count word frequencies and use them to set the lex.rank attribute."""
    freqs = Counter()
    oov_rank = 1
    vocab["<oov>"].rank = oov_rank
    vocab.lex_attr_getters[ID] = lambda word: oov_rank
    rank = 2
    for lex in vocab:
        lex.rank = oov_rank
    for doc in docs:
        assert doc.vocab is vocab
        for token in doc:
            lex = vocab[token.orth]
            freqs[lex.orth] += 1
    for word in force_include:
        lex = vocab[word]
        lex.rank = rank
        rank += 1
    for orth, count in freqs.most_common():
        lex = vocab[orth]
        if lex.text not in force_include:
            lex.rank = rank
            rank += 1
        if nTGT != 0 and rank >= nTGT:
            break
    output_docs = []
    for doc in docs:
        output_docs.append(Doc(vocab, words=[w.text for w in doc]))
        for token in output_docs[-1]:
            assert token.rank != 0, (token.text, token.vocab[token.text].rank)
    return output_docs

def get_dicts(vocab):
    '''
        Returns word2indx, indx2word
    '''
    word2indx = defaultdict(lambda: vocab['<oov'].rank)
    indx2word = defaultdict(lambda: '<oov>')
    for lex in vocab:
        word2indx[lex.text] = lex.rank
        indx2word[lex.rank] = lex.text
    return word2indx, indx2word



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
        '''
        Create a batch object from Xs, Ys pair
        Args:
            Xs_Ys:
                Xs: nB, nL1, nM
                Ys: nB, nL2, nM
        Returns Batch object:
            Xs: nB, nL, nM
            Ys: nB, nL, nM
            X_mask: nB, nL, nL
            y_mask: nB, nL, nL
        where nL = max(nL1, nL2)
        '''
        Xs, Ys = Xs_Ys
        nX = model.ops.asarray([x.shape[0] for x in Xs], dtype='i')
        nY = model.ops.asarray([y.shape[0] for y in Ys], dtype='i')

        nL = max(nX.max(), nY.max())
        Xs, unpad_dXs = pad_sequences(model.ops, Xs, pad_to=nL)
        Ys, unpad_dYs = pad_sequences(model.ops, Ys, pad_to=nL)

        def create_batch_backward(dXs_dYs, sgd=None):
            dXs, dYs = dXs_dYs
            dXs = unpad_dXs(dXs)
            dYs = unpad_dYs(dYs)
            return dXs, dYs

        batch = Batch((Xs, Ys), (nX, nY))
        # TODO: experiment
        return (batch.X, batch.X_mask, batch.y, batch.y_mask), create_batch_backward
        # return ((batch.X, batch.X_mask), (batch.y, batch.y_mask)), create_batch_backward
    model = layerize(create_batch_forward)
    return model


def pad_sequences(ops, seqs_in, pad_to=None):
    lengths = ops.asarray([len(seq) for seq in seqs_in], dtype='i')
    nB = len(seqs_in)
    if pad_to is None:
        pad_to = lengths.max()
    arr = ops.allocate((nB, pad_to) + seqs_in[0].shape[1:], dtype=seqs_in[0].dtype)
    for arr_i, seq in enumerate(seqs_in):
        arr[arr_i, :seq.shape[0]] = ops.asarray(seq)

    def unpad(padded):
        unpadded = [None] * len(lengths)
        for i in range(padded.shape[0]):
            unpadded[i] = padded[i, :lengths[i]]
        return unpadded
    return arr, unpad


def get_model_sentence(Yh, indx2word):
    '''
        Returns a sentence from the output of the projection layer
    '''
    sentences = []
    argmaxed = Model.ops.xp.argmax(Yh, axis=-1)
    for sentence in argmaxed:
        tokens = []
        for num in sentence:
            tokens.append(indx2word[num])
        sentences.append(tokens)
    return sentences

def get_loss(ops, Yh, Y_docs, Xmask):
    Y_ids = docs2ids(Y_docs)
    guesses = Yh.argmax(axis=-1)
    nC = Yh.shape[-1]
    Y = [to_categorical(y, nb_classes=nC) for y in Y_ids]
    nL = max(Yh.shape[1], max(y.shape[0] for y in Y))
    Y, _ = pad_sequences(ops, Y, pad_to=nL)
    is_accurate = (Yh.argmax(axis=-1) == Y.argmax(axis=-1))
    d_loss = Yh-Y
    for i, doc in enumerate(Y_docs):
        is_accurate[i, len(doc):] = 0
        d_loss[i, len(doc):] = 0
    return d_loss, is_accurate.sum()

def visualize(model, X0, Y0):
    rest_layers = model._layers[1:-1]
    groundwork = model._layers[0]
    last_layer = model._layers[-1]
    for layer in rest_layers:
        groundwork = chain(groundwork, layer)
    (X1, Xmask), (Y1, Ymask) = groundwork((X0, Y0))
    def get_padded_sentence(X, X1):
        length = X1[0].shape[0]
        sentX = X[0].text.split(' ')
        sentX.remove('')
        if len(sentX) > length:
            sentX = sentX[:length + 1]
        if len(sentX) < length:
            diff = length - len(sentX)
            pad = ['<pad>' for i in range(diff)]
            sentX.extend(pad)
        return sentX
    sentX = get_padded_sentence(X0, X1)
    sentY = get_padded_sentence(Y0, X1)
    last_layer([(X1, Xmask), (Y1, Ymask), (sentX, sentY)])

def FancyEmbed(width, rows, cols=(ORTH, SHAPE, PREFIX, SUFFIX)):
    from thinc.i2v import HashEmbed
    from thinc.v2v import Maxout
    from thinc.api import chain, concatenate
    tables = [HashEmbed(width, rows, column=i) for i in range(len(cols))]
    return chain(concatenate(*tables), Maxout(width, width*len(tables), pieces=3))



@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option", "nH", int),
    dropout=("model dropout", "option", "d", float),
    nS=('Number of encoders/decoder in the enc/dec stack.', "option", "nS", int),
    nB=('Batch size for the training', "option", "nB", int),
    nE=('Number of epochs for the training', "option", "nE", int),
    use_gpu=("Which GPU to use. -1 for CPU", "option", "g", int),
    lim=("Number of sentences to load from dataset", "option", "l", int),
    nM=("Embeddings size", "option", "nM", int),
    mL=("Max length sentence in dataset", "option", "mL", int),
    nTGT=("Vocabulary size", "option", "nTGT", int)


)
def main(nH=6, dropout=0.1, nS=6, nB=15, nE=20, use_gpu=-1, lim=2000,
        nM=300, mL=10, nTGT=1000):
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
    train_lim = min(lim, len(train_X))
    dev_lim = min(lim, len(dev_X))
    test_lim = min(lim, len(test_X))
    train_X, train_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                      train_X[:train_lim], train_Y[:train_lim], mL)
    dev_X, dev_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                  dev_X[:dev_lim], dev_Y[:dev_lim], mL)
    test_X, test_Y = spacy_tokenize(nlp_en.tokenizer, nlp_de.tokenizer,
                                    test_X[:test_lim], test_Y[:test_lim], mL)
    all_X_docs = train_X + dev_X + test_X
    all_y_docs = train_Y + dev_Y + test_Y
    train_X = set_numeric_ids(nlp_en.vocab, all_X_docs, nTGT=nTGT)
    train_Y = set_numeric_ids(nlp_de.vocab, all_y_docs, nTGT=nTGT)
    en_word2indx, en_indx2word = get_dicts(nlp_en.vocab)
    de_word2indx, de_indx2word = get_dicts(nlp_de.vocab)
    nTGT += 1

    with Model.define_operators({">>": chain}):
        embed_cols = [ORTH, SHAPE, PREFIX, SUFFIX]
        extractor = FeatureExtracter(attrs=embed_cols)
        position_encode = PositionEncode(mL, nM)
        model = (
            apply_layers(extractor, extractor)
            >> apply_layers(
                with_flatten(FancyEmbed(nM, 5000, cols=embed_cols)),
                with_flatten(FancyEmbed(nM, 5000, cols=embed_cols)),
            )
            >> apply_layers(Residual(position_encode), Residual(position_encode))
            >> create_batch()
            >> EncoderDecoder(nS=nS, nH=nH, nTGT=nTGT)
        )

    losses = [0.]
    train_accuracies = [0.]
    train_totals = [0.]
    dev_accuracies = [0.]
    dev_loss = [0.]
    def track_progress():
        correct = 0.
        total = 0.
        for batch in minibatch(zip(dev_X, dev_Y), size=1024):
            X, Y = zip(*batch)
            Yh, Y_mask = model((X, Y))
            L, C = get_loss(model.ops, Yh, Y, Y_mask)
            correct += C
            dev_loss[-1] += (L**2).sum()
            total += len(Y)
        dev_accuracies[-1] = correct / total
        n_train = train_totals[-1]
        print(len(losses), losses[-1], train_accuracies[-1]/n_train,
            dev_loss[-1], dev_accuracies[-1])
        dev_loss.append(0.)
        losses.append(0.)
        train_accuracies.append(0.)
        dev_accuracies.append(0.)
        train_totals.append(0.)

    with model.begin_training(batch_size=nB, nb_epoch=nE) as (trainer, optimizer):
        trainer.dropout = dropout
        trainer.dropout_decay = 1e-4
        optimizer.alpha = 0.001
        optimizer.L2 = 1e-6
        optimizer.max_grad_norm = 1.0
        trainer.each_epoch.append(track_progress)
        optimizer.alpha = 0.001
        optimizer.L2 = 1e-6
        optimizer.max_grad_norm = 1.0
        for X, Y in trainer.iterate(train_X, train_Y):
            # TODO: fix this
            # (Yh, X_mask), backprop = model.begin_update((X, Y), drop=dropout)
            (Yh, X_mask), backprop = model.begin_update((X, Y))
            dYh, C = get_loss(model.ops, Yh, Y, X_mask)
            backprop(dYh, sgd=optimizer)
            losses[-1] += (dYh**2).sum()
            train_accuracies[-1] += C
            train_totals[-1] += sum(len(y) for y in Y)

    # visualize(model, [train_X[0]], [train_Y[0]])
    # visualize(model, [train_X[1]], [train_Y[1]])
    # visualize(model, [train_X[2]], [train_Y[2]])
    # visualize(model, [train_X[3]], [train_Y[3]])
    # visualize(model, [train_X[4]], [train_Y[4]])

if __name__ == '__main__':
    plac.call(main)
