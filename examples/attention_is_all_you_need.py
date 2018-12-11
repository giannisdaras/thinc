from thinc.neural.ops import NumpyOps, CupyOps, Ops
from thinc.neural.optimizers import initNoAm
from thinc.neural.util import add_eos_bos
from thinc.linear.linear import LinearModel
from thinc.v2v import Model
import numpy as np
import plac
import spacy
from thinc.extra.datasets import get_iwslt
from spacy.lang.en import English

class ModelException(Exception):
    pass

class BatchesException(Exception):
    pass

@plac.annotations(
    heads=("number of heads of the multiheaded attention", "option"),
    dropout=("model dropout", "option")
)
def main(heads=6, dropout=0.1):
    if (CupyOps.xp != None):
        Model.ops = CupyOps()
        Model.Ops = CupyOps
        print('Training on GPU')
    else:
        print('Training on CPU')
    train, dev, test = get_iwslt()
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    test_X, test_Y = zip(*test)
    nlp = spacy.load('en_core_web_sm')
    tokenizer = English().Defaults.create_tokenizer(nlp)
    train_X = [doc.text.split(' ') for doc in tokenizer.pipe(train_X[:20])]
    train_Y = [doc.text.split(' ') for doc in tokenizer.pipe(train_Y[:20])]


    ''' Mark Y sentences '''
    train_Y = add_eos_bos(train_Y)
    model = LinearModel(2)
    with model.begin_training(train_X, train_Y, batch_size=2, nb_epoch=1) as (trainer, optimizer):
        for X, y, X_mask, y_mask in trainer.batch_mask(train_X, train_Y):
            pass
            # next step vectorizer



    # vectorizer = spacy.load('en_vectors_web_lg')
    # ''' Mark Y sentences '''
    # train_Y, dev_Y, test_Y = add_eos_bos(train_Y), \
    #     add_eos_bos(dev_Y), add_eos_bos(test_Y)
    #
    # ''' batchify and mask '''
    # train_X, train_Y, X_mask, Y_mask = batchify_and_mask(Model.ops)
    # with model.begin_training(train_X, train_Y, optimizer=initNoAm(model.nI)) \
    #         as (trainer, optimizer):
    #         trainer.each_epoch(append(lambda: print(model.evaluate(dev_X, dev_Y))))
    #         with X, y in trainer.iterate(train_X, train_Y):
    #             ''' at this stage, we have X, y batched but not padded, so we
    #             are missing padding and input/output masks.
    #             Also, the representation is token-level, not embedding level,
    #             meaning that we need an extra vectorizer step.
    #             '''
    #             yh, backprop = model.begin_update(X, drop=trainer.dropout)





if __name__ == '__main__':
    plac.call(main)
