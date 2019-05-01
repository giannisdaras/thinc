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
    PositionEncode, docs2ids, FancyEmbed
from bert_textcat import create_model_input, get_mask
from thinc.misc import Residual
from thinc.v2v import Model
from thinc.neural._classes.encoder_decoder import Encoder
from thinc.extra.datasets import get_iwslt

random.seed(0)
numpy.random.seed(0)


def random_mask(X0, nlp, indx2word, vocab, mL):
    nC = int(0.15 * max([len(x) for x in X0]))
    indices = \
        [Model.ops.xp.random.randint(0, len(x), nC) for x in X0]
    docs = []
    for sent_indx in range(len(X0)):
        words = [w.text for w in X0[sent_indx]]
        for word in words:
            dice = int(Model.ops.xp.random.randint(1, 11, 1))
            if dice <= 8:
                word = '<mask>'
            elif dice == 9:
                vocab_indx = \
                    int(Model.ops.xp.random.randint(0, len(nlp.vocab), 1))
                random_word = indx2word[vocab_indx]
                word = random_word
        docs.append(Doc(vocab, words=words))
    return docs, indices


def spacy_tokenize(X_tokenizer, X, mL=50):
    X_out = []
    for x in X:
        xdoc = X_tokenizer('<cls> ' + x.strip())
        if len(xdoc) < mL:
            X_out.append(xdoc)
    return X_out


def set_rank(vocab, docs, force_include=("<oov>", "<eos>", "<bos>",
             "<cls>", "<mask>"), nTGT=5000):
    ''' A function to prepare vocab ids '''
    freqs = Counter()

    # set oov rank
    oov_rank = 1
    vocab["<oov>"].rank = oov_rank
    vocab.lex_attr_getters[ID] = lambda word: oov_rank

    # set all words to oov
    for lex in vocab:
        lex.rank = oov_rank
    rank = 2

    # set ids for the special tokens
    for word in force_include:
        lex = vocab[word]
        lex.rank = rank
        rank += 1

    # count frequencies of orth
    for doc in docs:
        assert doc.vocab is vocab
        for token in doc:
            lex = vocab[token.orth]
            freqs[lex.orth] += 1

    # update the ids of the most commont nTGT words
    for orth, count in freqs.most_common():
        lex = vocab[orth]
        if lex.text not in force_include:
            lex.rank = rank
            rank += 1
        if nTGT != 0 and rank >= nTGT:
            break


def set_numeric_ids(vocab, docs):
    output_docs = []
    for doc in docs:
        output_docs.append(Doc(vocab, words=[w.text for w in doc]))
        for token in output_docs[-1]:
            assert token.rank != 0, (token.text, token.vocab[token.text].rank)
    return output_docs


def get_loss(Xh, X_docs, indices):
    X_ids = docs2ids(X_docs)
    nb_classes = Xh.shape[-1]
    X = [to_categorical(y, nb_classes=nb_classes) for y in X_ids]
    X, _ = pad_sequences(Model.ops, X)

    ''' Loss calculation '''
    import pdb; pdb.set_trace()
    dXh = Model.ops.xp.zeros(Xh.shape)
    return dXh, is_accurate.sum(), is_accurate.sum() + is_not_accurate.sum()


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
def main(nH=6, dropout=0.0, nS=6, nB=32, nE=20, use_gpu=-1, lim=2000,
         nM=300, mL=100, save=False, nTGT=5000, save_name="model.pkl"):
    if use_gpu != -1:
        # TODO: Make specific to different devices, e.g. 1 vs 0
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
            >> with_flatten(FancyEmbed(nM, 5000, cols=embed_cols))
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

        def track_progress():
            correct = 0.
            total = 0.
            for X0 in minibatch(dev, size=1024):
                X1, indices = random_mask(X0, nlp, indx2word, mL)
                Xh = model(X1)
                L, C, total = get_loss(Xh, X0, indices)
                correct += C
                dev_loss[-1] += (L**2).sum()
            dev_accuracies[-1] = correct / total
            n_train = train_totals[-1]
            print(len(losses), losses[-1], train_accuracies[-1]/n_train,
                  dev_loss[-1], dev_accuracies[-1])
            dev_loss.append(0.)
            losses.append(0.)
            train_accuracies.append(0.)
            dev_accuracies.append(0.)
            train_totals.append(0.)

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
                X1, indices = random_mask(X0, nlp, indx2word, nlp.vocab, mL)
                Xh, backprop = model.begin_update(X1)
                dXh, C, total = get_loss(Xh, X0, indices)
                backprop(dXh, sgd=optimizer)
                losses[-1] += (dYh**2).sum()
                train_accuracies[-1] += C
                train_totals[-1] += total
        if save:
            model.to_disk(save_name)





if __name__ == '__main__':
    plac.call(main)
