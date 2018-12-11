from thinc.neural.ops import NumpyOps, CupyOps, Ops
from thinc.neural.optimizers import initNoAm
from thinc.neural.util import add_eos_bos
from thinc.v2v import Model
import numpy as np
import plac
import spacy
from thinc.extra.datasets import get_iwslt


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
    tokenizer = lambda x: spacy.load('en_core_web_sm').tokenizer(x)
    vectorizer = spacy.load('en_vectors_web_lg')
    ''' Tokenize data '''
    train_X, train_Y = [tok.text for text in tokenizer(train_X)], \
                       [tok.text for text in tokenizer(train_Y)]
    dev_X, dev_Y = [tok.text for text in tokenizer(dev_X)], \
                   [tok.text for text in tokenizer(dev_Y)]
    test_X, test_Y = [tok.text for text in tokenizer(test_X)] \
                     [tok.text for text in tokenizer(test_Y)]
    ''' Mark Y sentences '''
    train_Y, dev_Y, test_Y = add_eos_bos(train_Y), \
        add_eos_bos(dev_Y), add_eos_bos(test_Y)

    raise ModelException('Model not composed yet.')
    with model.begin_training(train_X, train_Y, optimizer=initNoAm(model.nI)) \
            as (trainer, optimizer):
            trainer.each_epoch(append(lambda: print(model.evaluate(dev_X, dev_Y))))
            with X, y in trainer.iterate(train_X, train_Y):
                ''' at this stage, we have X, y batched but not padded, so we
                are missing padding and input/output masks.
                Also, the representation is token-level, not embedding level,
                meaning that we need an extra vectorizer step.
                '''
                yh, backprop = model.begin_update(X, drop=trainer.dropout)





if __name__ == '__main__':
    plac.call(main)
