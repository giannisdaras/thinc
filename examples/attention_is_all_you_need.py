from thinc.neural.ops import CupyOps
from thinc.neural.util import add_eos_bos, subsequent_mask
from thinc.neural.util import numericalize, numericalize_vocab
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.static_vectors import StaticVectors
from thinc.v2v import Model
import plac
import spacy
from thinc.extra.datasets import get_iwslt
from spacy.lang.en import English
from spacy.lang.de import German
import numpy as np
import pdb
MODEL_SIZE = 300


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
    batch_size=('Batch size for the training', "option")
)
def main(heads=6, dropout=0.1, stack=6, batch_size=2):
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
    X_positions = Model.ops.position_encode(50, MODEL_SIZE)
    print('Position encodings computed successfully')
    ''' Read dataset '''
    nlp_en = spacy.load('en_core_web_md')
    nlp_de = spacy.load('de_core_news_md')
    eos_vector = Model.ops.xp.random.rand(1, MODEL_SIZE)
    bos_vector = Model.ops.xp.random.rand(1, MODEL_SIZE)
    nlp_de.vocab.set_vector('<eos>', eos_vector)
    nlp_de.vocab.set_vector('<bos>', bos_vector)
    en_tokenizer = English().Defaults.create_tokenizer(nlp_en)
    de_tokenizer = German().Defaults.create_tokenizer(nlp_de)
    train_X, dev_X, test_X = spacy_tokenize(en_tokenizer, train_X[:20],
                                            dev_X[:10], test_X[:10])
    train_Y, dev_Y, test_Y = spacy_tokenize(de_tokenizer, train_Y[:20],
                                            dev_Y[:10], test_Y[:10])
    en_embeddings = StaticVectors('en_core_web_md', MODEL_SIZE, column=0)
    de_embeddings = StaticVectors('de_core_news_md', MODEL_SIZE, column=0)
    en_word2indx, en_indx2word = numericalize_vocab(nlp_en)
    de_word2indx, de_indx2word = numericalize_vocab(nlp_de)
    with model.begin_training(train_X, train_Y, batch_size=2, nb_epoch=1) as \
            (trainer, optimizer):
        ''' add beginning and ending of sentence marks '''
        train_Y = add_eos_bos(train_Y)
        for X, y, X_mask, y_mask in trainer.batch_mask(train_X, train_Y):
            ''' numericalize text '''
            for indx, _x in enumerate(X):
                X[indx] = numericalize(en_word2indx, _x)
            for indx, _y in enumerate(y):
                y[indx] = numericalize(de_word2indx, _y)
            ''' get text embeddings '''
            X = en_embeddings(Model.ops.asarray(X))
            y = de_embeddings(Model.ops.asarray(y))
            sentence_size = X.shape[1]
            ''' add position encodings '''
            X = X + X_positions[:sentence_size]
            y = y + X_positions[:sentence_size]
            batch = Batch(X, y, X_mask, y_mask)
            yh, backprop = model.begin_update(batch, drop=trainer.dropout)


if __name__ == '__main__':
    plac.call(main)
