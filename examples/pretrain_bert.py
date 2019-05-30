''' Pretrain bert in the language model task '''
from __future__ import unicode_literals
from collections import defaultdict
import random
import numpy
import plac
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.attrs import ID, ORTH, SHAPE, PREFIX, SUFFIX
from thinc.neural.util import to_categorical, minibatch
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.softmax import Softmax
from thinc.misc import FeatureExtracter
from thinc.api import wrap, chain, with_flatten, with_reshape
from attention_is_all_you_need import get_dicts, pad_sequences, \
    PositionEncode, docs2ids, FancyEmbed, set_rank, set_numeric_ids
from bert_textcat import create_model_input, get_mask
from thinc.misc import Residual
from thinc.v2v import Model
from thinc.neural._classes.encoder_decoder import Encoder
from thinc.extra.datasets import get_iwslt


def random_mask(X0, nlp, indx2word, vocab, mL):
    nB = len(X0)
    nL = max([len(x) for x in X0])
    nC = int(0.15 * nL)
    # produce masked tokens indices
    indices = \
        [Model.ops.xp.random.randint(0, len(x), nC) for x in X0]
    ''' We need to compute loss only for the masked tokens '''
    loss_mask = Model.ops.xp.zeros((nB, nL))
    docs = []
    for sent_indx in range(nB):
        words = [w.text for w in X0[sent_indx]]
        new_words = []
        for i, word in enumerate(words):
            dice = int(Model.ops.xp.random.randint(1, 11, 1))
            if i not in indices[sent_indx]:
                new_words.append(word)
            else:
                loss_mask[sent_indx, i] = 1.0
                if dice == 10:
                    new_words.append(word)
                elif dice <= 8:
                    new_words.append('<mask>')
                else:
                    ''' Choose a random word from the vocabulary '''
                    vocab_indx = \
                        int(Model.ops.xp.random.randint(0, len(nlp.vocab), 1))
                    random_word = indx2word[vocab_indx]
                    new_words.append(random_word)
        docs.append(Doc(vocab, words=new_words))
    return docs, loss_mask


def spacy_tokenize(X_tokenizer, X, mL=50):
    X_out = []
    for x in X:
        xdoc = X_tokenizer('<cls> ' + x.strip())
        if len(xdoc) < mL:
            X_out.append(xdoc)
    return X_out


def get_loss(Xh, original_docs, masked_docs, loss_mask):
    ''' Calculate loss '''

    ''' Convert original docs to (nB, nL, nTGT) with one hot encoding '''
    X_ids = docs2ids(original_docs)
    nb_classes = Xh.shape[-1]
    X = [to_categorical(y, nb_classes=nb_classes) for y in X_ids]
    X, _ = pad_sequences(Model.ops, X)

    ''' Loss calculation only on the masked positions '''
    dXh = Xh - X
    dXh = dXh * Model.ops.xp.expand_dims(loss_mask, axis=2)

    ''' Calculate number of accurate, inaccurate tokens '''
    accurate = Xh.argmax(axis=-1) == X.argmax(axis=-1)
    inaccurate = Xh.argmax(axis=-1) != X.argmax(axis=-1)
    accurate = accurate * loss_mask
    inaccurate = inaccurate * loss_mask

    return dXh, accurate.sum(), accurate.sum() + inaccurate.sum()


@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option", "nH", int),
    dropout=("model dropout", "option", "d", float),
    nS=('Number of encoders in the enc stack.', "option", "nS", int),
    nB=('Batch size for the training', "option", "nB", int),
    nE=('Number of epochs for the training', "option", "nE", int),
    use_gpu=("Which GPU to use. -1 for CPU", "option", "g", int),
    lim=("Number of sentences to load from dataset", "option", "l", int),
    nM=("Embeddings size", "option", "nM", int),
    nTGT=("Vocabulary size", "option", "nTGT", int),
    mL=("Max length sentence in dataset", "option", "mL", int),
    save=("Save model to disk", "option", "save", bool),
    save_name=("Name of file saved to disk. Save option must be enabled")
)
def main(nH=6, dropout=0.0, nS=6, nB=32, nE=20, use_gpu=-1, lim=1000000,
         nM=300, mL=100, save=False, nTGT=5000, save_name="model.pkl"):
    if use_gpu != -1:
        spacy.require_gpu()
        device = 'cuda'
    else:
        device = 'cpu'

    ''' Read dataset '''
    nlp = spacy.load('en_core_web_sm')
    print('English model loaded')
    for control_token in ("<eos>", "<bos>", "<pad>", "<cls>", "<mask>"):
        nlp.tokenizer.add_special_case(control_token, [{ORTH: control_token}])

    train, dev, test = get_iwslt()
    print('Dataset loaded')

    train, _ = zip(*train)
    dev, _ = zip(*dev)
    test, _ = zip(*test)

    train = train[:lim]
    dev = dev[:lim]
    test = test[:lim]

    ''' Tokenize '''
    train = spacy_tokenize(nlp.tokenizer, train, mL=mL)
    dev = spacy_tokenize(nlp.tokenizer, dev, mL=mL)
    test = spacy_tokenize(nlp.tokenizer, test, mL=mL)
    print('Tokenization finished')

    ''' Set rank based on all the docs '''
    all_docs = train + dev + test
    set_rank(nlp.vocab, all_docs, nTGT=nTGT)

    train = set_numeric_ids(nlp.vocab, train)
    dev = set_numeric_ids(nlp.vocab, dev)
    test = set_numeric_ids(nlp.vocab, test)
    print('Numeric ids set')

    word2indx, indx2word = get_dicts(nlp.vocab)
    print('Vocab dictionaries grabbed')

    with Model.define_operators({">>": chain}):
        embed_cols = [ORTH, SHAPE, PREFIX, SUFFIX]
        extractor = FeatureExtracter(attrs=embed_cols)
        position_encode = PositionEncode(mL, nM)
        model = (
            FeatureExtracter(attrs=embed_cols)
            >> with_flatten(FancyEmbed(nM, nTGT, cols=embed_cols))
            >> Residual(position_encode)
            >> create_model_input()
            >> Encoder(nM=nM, nS=nS, nH=nH, device=device)
            >> with_reshape(Softmax(nO=nTGT, nI=nM))
        )
        ''' Progress tracking '''
        losses = [0.]
        train_accuracies = [0.]
        train_totals = [0.]
        dev_accuracies = [0.]
        dev_loss = [0.]
        epoch_ind = 0

        def track_progress():
            correct = 0.
            total = 0.
            ''' Get dev stats '''
            for X0 in minibatch(dev, size=nB):
                X1, loss_mask = random_mask(X0, nlp, indx2word, nlp.vocab, mL)
                Xh = model(X1)
                L, C, t = get_loss(Xh, X0, X1, loss_mask)
                correct += C
                total += t
                dev_loss[-1] += (L**2).sum()
            dev_accuracies[-1] = correct / total
            print(len(losses), losses[-1], train_accuracies[-1] / train_totals[-1],
                  dev_loss[-1], dev_accuracies[-1])
            dev_loss.append(0.)
            losses.append(0.)
            train_accuracies.append(0.)
            dev_accuracies.append(0.)
            train_totals.append(0.)
            epoch_ind += 1
            if save:
                model.to_disk('.models/' + epoch_ind + save_name)

        ''' Model training '''
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
            for X0, _ in trainer.iterate(train, train):
                X1, loss_mask = random_mask(X0, nlp, indx2word, nlp.vocab, mL)
                Xh, backprop = model.begin_update(X1)
                dXh, C, total = get_loss(Xh, X0, X1, loss_mask)
                backprop(dXh, sgd=optimizer)
                losses[-1] += (dXh**2).sum()
                train_accuracies[-1] += C
                train_totals[-1] += total
        if save:
            model.to_disk('.models/' + save_name)


if __name__ == '__main__':
    plac.call(main)
