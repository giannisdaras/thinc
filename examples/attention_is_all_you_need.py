from thinc.neural.ops import CupyOps
import pickle
import torch.nn as nn
import torch
import gensim.models.keyedvectors as word2vec
from thinc.neural.util import add_eos_bos
from thinc.neural.util import numericalize, numericalize_vocab
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.static_vectors import StaticVectors
from thinc.v2v import Model
import plac
import spacy
from thinc.extra.datasets import get_iwslt
from spacy.lang.en import English
from spacy.lang.de import German
import pdb
MODEL_SIZE = 300
debug = True

def save_object(obj, filename):
    ''' function to save dataset as pickle object '''
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    ''' load pickle with filename '''
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return p


class Batch:
    def __init__(self, X, y, X_mask, y_mask):
        self.X = X
        self.y = y
        self.X_mask = X_mask
        self.y_mask = y_mask
        self.batch_size = X.shape[0]


def spacy_tokenize(tokenizer, *args):
    result = []
    for data in args:
        result.append([doc.text.split(' ') for doc in tokenizer.pipe(data)])
    return result


@plac.annotations(
    heads=("number of heads of the multiheaded attention", "option"),
    dropout=("model dropout", "option"),
    stack=('Number of encoders/decoder in the enc/dec stack.', "option"),
    german_model=('Path to german model', "option")
)
def main(heads=6, dropout=0.1, stack=6, german_model=
         '/home/giannis/vectors/gensim_german_model'):
    if (CupyOps.xp is not None):
        Model.ops = CupyOps()
        Model.Ops = CupyOps
        print('Running on GPU')
    else:
        print('Running on CPU')
    train, dev, test = get_iwslt()
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    test_X, test_Y = zip(*test)
    model = EncoderDecoder()
    if debug:
        with model.begin_training(train_X, train_Y, batch_size=2, nb_epoch=1) \
                                 as (trainer, optimizer):
            X = load_object('X.pickle')
            y = load_object('y.pickle')
            X_mask = load_object('X_mask.pickle')
            y_mask = load_object('y_mask.pickle')
            batch = Batch(X, y, X_mask, y_mask)
            model.begin_update(batch)
    else:
        X_positions = Model.ops.position_encode(50, MODEL_SIZE)
        print('Positions encodings computed successfully')
        ''' Read dataset '''
        nlp_en = spacy.load('en_core_web_lg')
        nlp_de = spacy.load('de_core_news_sm')
        en_tokenizer = English().Defaults.create_tokenizer(nlp_en)
        de_tokenizer = German().Defaults.create_tokenizer(nlp_de)
        train_X, dev_X, test_X = spacy_tokenize(en_tokenizer, train_X[:20],
                                                dev_X[:10], test_X[:10])
        train_Y, dev_Y, test_Y = spacy_tokenize(de_tokenizer, train_Y[:20],
                                                dev_Y[:10], test_Y[:10])
        en_embeddings = StaticVectors('en_core_web_lg', MODEL_SIZE, column=0)
        de_vectors = word2vec.KeyedVectors.\
            load_word2vec_format(german_model, binary=True).vectors
        de_embeddings = nn.Embedding(len(de_vectors), MODEL_SIZE)
        de_embeddings.weight.data.copy_(torch.from_numpy(de_vectors))
        en_word2indx, en_indx2word = numericalize_vocab(nlp_en)
        de_word2indx, de_indx2word = numericalize_vocab(nlp_de, rank=False)
        print('Embeddings loaded successfully')
        with model.begin_training(train_X, train_Y, batch_size=2, nb_epoch=1) as \
                (trainer, optimizer):
            for X, y, X_mask, y_mask in trainer.batch_mask(train_X, train_Y):
                for indx, _x in enumerate(X):
                    X[indx] = numericalize(en_word2indx, _x)
                for indx, _y in enumerate(y):
                    y[indx] = numericalize(de_word2indx, _y)
                X = en_embeddings(Model.ops.asarray(X))
                y = Model.ops.asarray(de_embeddings(torch.tensor(y)).detach().squeeze().numpy())
                sentence_size = X.shape[1]
                X = X + X_positions[:sentence_size]
                y = y + X_positions[:sentence_size]
                save_object(X, 'X.pickle')
                save_object(y, 'y.pickle')
                save_object(X_mask, 'X_mask.pickle')
                save_object(y_mask, 'y_mask.pickle')
                batch = Batch(X, y, X_mask, y_mask)
                yh, backprop = model.begin_update(batch, drop=trainer.dropout)


if __name__ == '__main__':
    plac.call(main)
