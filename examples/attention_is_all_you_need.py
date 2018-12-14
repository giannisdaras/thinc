from thinc.neural.ops import NumpyOps, CupyOps, Ops
from thinc.neural.optimizers import initNoAm
from thinc.neural.util import add_eos_bos
from thinc.linear.linear import LinearModel
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.v2v import Model
from thinc.api import chain, clone
import numpy as np
import plac
import spacy
from thinc.extra.datasets import get_iwslt
from spacy.lang.en import English
import pdb

MODEL_SIZE = 300


class Batch:
    def __init__(self, X, y, X_mask, y_mask):
        self.X = X
        self.y = y
        self.X_mask = X_mask
        self.y_mask = y_mask
        self.batch_size = X.shape[0]


def vectorize(batch, vectorizer):
    ''' Vectorize a batch of tokens with a vectorizer nlp object
    A batch (non-vectorized) contains some sentences (s) and some tokens (t).
    A vectorized batch contains some sentences(s) and the d-dimensional vectorized
    version of the tokens (d).
    '''
    nB = len(batch)
    nT = len(batch[0])
    nD = len(vectorizer.vocab['test'].vector)
    vectorized_batch = Model.ops.xp.empty([nB, nT, nD])
    for sent_idx, sent in enumerate(batch):
        for token_idx, token in enumerate(sent):
            vectorized_token = Model.ops.asarray(vectorizer.vocab[token].vector)
            vectorized_batch[sent_idx, token_idx, :] = vectorized_token
    return Model.ops.asarray(vectorized_batch)


@plac.annotations(
    heads=("number of heads of the multiheaded attention", "option"),
    dropout=("model dropout", "option"),
    stack=('Number of encoders/decoder in the enc/dec stack.', "option")
)
def main(heads=6, dropout=0.1, stack=6):
    if (CupyOps.xp != None):
        Model.ops = CupyOps()
        Model.Ops = CupyOps
        print('Running on GPU')
    else:
        print('Running on CPU')
    train, dev, test = get_iwslt()
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    test_X, test_Y = zip(*test)
    nlp = spacy.load('en_core_web_sm')
    tokenizer = English().Defaults.create_tokenizer(nlp)
    train_X = [doc.text.split(' ') for doc in tokenizer.pipe(train_X[:20])]
    train_Y = [doc.text.split(' ') for doc in tokenizer.pipe(train_Y[:20])]
    vectorizer = spacy.load('en_vectors_web_lg')

    ''' Mark Y sentences '''
    train_Y = add_eos_bos(train_Y)
    model = EncoderDecoder()
    with model.begin_training(train_X, train_Y, batch_size=2, nb_epoch=1) as (trainer, optimizer):
        for X, y, X_mask, y_mask in trainer.batch_mask(train_X, train_Y):
            X, y = vectorize(X, vectorizer), vectorize(y, vectorizer)
            batch_size = X.shape[0]
            max_sentence_in_batch = X.shape[1]
            model_size = X.shape[2]
            ''' there is a computational overhead here. Positional encodings
            can be get all at once, but let's keep it simple for now '''
            X_positions = model.ops.position_encode(max_sentence_in_batch, model_size)
            X = X + X_positions
            y = y + X_positions
            batch = Batch(X, y, X_mask, y_mask)
            yh, backprop = model.begin_update(batch, drop=trainer.dropout)
            backprop(yh, optimizer)




if __name__ == '__main__':
    plac.call(main)
