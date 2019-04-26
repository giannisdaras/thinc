''' Text categorization using the encoder stack '''
from __future__ import unicode_literals
from collections import defaultdict
import random
import numpy
import plac
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, ORTH, SHAPE, PREFIX, SUFFIX
from thinc.extra.datasets import imdb
from thinc.loss import categorical_crossentropy
from spacy._ml import link_vectors_to_models
from thinc.neural.util import to_categorical, minibatch
from thinc.neural._classes.encoder_categorizer import Categorizer
from thinc.neural._classes.embed import Embed
from thinc.misc import FeatureExtracter
from thinc.api import wrap, chain, with_flatten, layerize
from thinc.misc import Residual
from thinc.v2v import Model
import numpy.random
import random
import pickle
import sys

random.seed(0)
numpy.random.seed(0)


def get_mask(X, nX):
    nB = X.shape[0]
    nL = X.shape[1]
    X_mask = Model.ops.allocate((nB, nL, nL))
    for i, length in enumerate(nX):
        X_mask[i, :, :length] = 1.0
    return X_mask


def spacy_tokenize(X_tokenizer, X, mL=50):
    X_out = []
    for x in X:
        xdoc = X_tokenizer('<cls>' + x.strip())
        if len(xdoc) < mL:
            X_out.append(xdoc)
    return X_out


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


def set_numeric_ids(vocab, docs, force_include=("<oov>", "<eos>", "<bos>", "<cls>")):
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


def create_model_input():
    def create_model_input_forward(Xs, drop=0.):
        nX = model.ops.asarray([x.shape[0] for x in Xs], dtype='i')
        nL = nX.max()
        Xs, unpad_dXs = pad_sequences(model.ops, Xs, pad_to=nL)
        X_mask = get_mask(Xs, nX)

        def create_model_input_backward(dXs, sgd=None):
            dXs = unpad_dXs(dXs)
            return dXs

        return (Xs, X_mask), create_model_input_backward
    model = layerize(create_model_input_forward)
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


def get_loss(Yh, Y):
    Y = [to_categorical(y, nb_classes=2) for y in Y]
    Y = Model.ops.xp.asarray(Y)[:, 0, :]
    is_accurate = (Yh.argmax(axis=-1) == Y.argmax(axis=-1))
    dYh = Yh - Y
    return dYh, is_accurate.sum()


def FancyEmbed(width, rows, cols=(ORTH, SHAPE, PREFIX, SUFFIX)):
    from thinc.i2v import HashEmbed
    from thinc.v2v import Maxout
    from thinc.api import chain, concatenate
    tables = [HashEmbed(width, rows, column=i) for i in range(len(cols))]
    return chain(concatenate(*tables), Maxout(width, width*len(tables), pieces=3))


def transform_data(Y):
    Y_ = []
    for i in Y:
        if i == 0:
            Y_.append(["NEGATIVE"])
        elif i == 1:
            Y_.append(["POSITIVE"])
    return Y_

@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option", "nH", int),
    dropout=("model dropout", "option", "d", float),
    nS=('Number of encodersin the enc stack.', "option", "nS", int),
    nB=('Batch size for the training', "option", "nB", int),
    nE=('Number of epochs for the training', "option", "nE", int),
    use_gpu=("Which GPU to use. -1 for CPU", "option", "g", int),
    lim=("Number of sentences to load from dataset", "option", "l", int),
    nM=("Embeddings size", "option", "nM", int),
    mL=("Max length sentence in dataset", "option", "mL", int),
    save=("Save model to disk", "option", "save", bool),
    save_name=("Name of file saved to disk. Save option must be enabled")
)
def main(nH=6, dropout=0.0, nS=6, nB=32, nE=20, use_gpu=-1, lim=2000,
        nM=300, mL=100, save=False, save_name="model.pkl"):
    if use_gpu != -1:
        # TODO: Make specific to different devices, e.g. 1 vs 0
        spacy.require_gpu()

    ''' Read dataset '''
    nlp = spacy.load('en_core_web_sm')
    for control_token in ("<eos>", "<bos>", "<pad>", "<cls>"):
        nlp.tokenizer.add_special_case(control_token, [{ORTH: control_token}])
    train, dev = imdb()
    print('Loaded imdb dataset')
    train = train[:lim]
    dev = dev[:lim]
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    train_X = spacy_tokenize(nlp.tokenizer, train_X, mL=mL)
    dev_X = spacy_tokenize(nlp.tokenizer, dev_X, mL=mL)
    print('Tokenized dataset')
    train_X = set_numeric_ids(nlp.vocab, train_X)
    dev_X = set_numeric_ids(nlp.vocab, dev_X)
    print('Numeric ids ready')
    with Model.define_operators({">>": chain}):
        embed_cols = [ORTH, SHAPE, PREFIX, SUFFIX]
        extractor = FeatureExtracter(attrs=embed_cols)
        position_encode = PositionEncode(mL, nM)
        model = (
            FeatureExtracter(attrs=embed_cols)
            >> with_flatten(FancyEmbed(nM, 5000, cols=embed_cols))
            >> Residual(position_encode)
            >> create_model_input()
            >> Categorizer(nM=nM, nS=nS, nH=nH)
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
            Yh = model(X)
            total = len(X)
            L, C = get_loss(Yh, Y)
            correct += C
            dev_loss[-1] += (L**2).sum()
            total += len(X)
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
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
        for X, Y in trainer.iterate(train_X, train_Y):
            Yh, backprop = model.begin_update(X)
            total = len(X)
            dYh, C = get_loss(Yh, Y)
            backprop(dYh, sgd=optimizer)
            losses[-1] += (dYh**2).sum()
            train_accuracies[-1] += C
            train_totals[-1] += len(Y)
    if save:
        model.to_disk(save_name)



if __name__ == '__main__':
    plac.call(main)
