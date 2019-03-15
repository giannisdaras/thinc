from thinc.neural.ops import CupyOps
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
import pickle
import pdb
MODEL_SIZE = 300


class Batch:
    ''' Objects of this class share pairs X,Y along with their
        padding/future-words-hiding masks '''
    def __init__(self, pair, pad_masks, lengths):
        # X, y: nB x nL x nM
        # X_mask, y_pad_mask: nB x nL
        # y_mask: nB x nL x nL
        X, y = pair
        self.X_mask, self.y_pad_mask = pad_masks
        nX, nY = lengths
        self.X = X
        self.y = y
        self.nB = X.shape[0]
        self.nL = X.shape[1]
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
        self.y_mask = Model.ops.xp.expand_dims(self.y_pad_mask, -2) & \
            self.subsequent_mask(self.nL)

    def subsequent_mask(self, nL):
        return (Model.ops.xp.triu(Model.ops.xp.ones([1, nL, nL]), k=1) == 0)


def spacy_tokenize(tokenizer, *args):
    result = []
    for data in args:
        result.append([doc.text.split(' ') for doc in tokenizer.pipe(data)])
    return result


@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option"),
    dropout=("model dropout", "option"),
    nS=('Number of encoders/decoder in the enc/dec stack.', "option"),
    nB=('Batch size for the training', "option"),
    nE=('Number of epochs for the training', "option")
)
def main(nH=6, dropout=0.1, nS=6, nB=2, nE=1):
    if (CupyOps.xp is not None):
        Model.ops = CupyOps()
        Model.Ops = CupyOps
        print('Running on GPU')
    else:
        print('Running on CPU')

    # DEBUG MODE:
    # model = EncoderDecoder()
    # X = Model.ops.xp.random.rand(2, 17, 300)
    # y = Model.ops.xp.random.rand(2, 17, 300)
    # batch = Batch(X, y, None, None)
    # yh, backprop = model.begin_update(batch)
    # _1, _2, _3 = yh.shape
    # print(backprop(Model.ops.xp.random.rand(_1, _2, _3, dtype=Model.ops.xp.float32)))
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
    print('Models loaded')
    eos_vector = Model.ops.xp.random.rand(MODEL_SIZE)
    bos_vector = Model.ops.xp.random.rand(MODEL_SIZE)
    nlp_de.vocab.set_vector('<eos>', eos_vector)
    nlp_de.vocab.set_vector('<bos>', bos_vector)
    en_tokenizer = English().Defaults.create_tokenizer(nlp_en)
    de_tokenizer = German().Defaults.create_tokenizer(nlp_de)
    train_X, dev_X, test_X = spacy_tokenize(en_tokenizer, train_X[:20],
                                            dev_X[:10], test_X[:10])
    train_Y, dev_Y, test_Y = spacy_tokenize(de_tokenizer, train_Y[:20],
                                            dev_Y[:10], test_Y[:10])
    train_Y = add_eos_bos(train_Y)
    en_embeddings = StaticVectors('en_core_web_md', MODEL_SIZE, column=0)
    de_embeddings = StaticVectors('de_core_news_md', MODEL_SIZE, column=0)
    en_word2indx, en_indx2word = numericalize_vocab(nlp_en)
    de_word2indx, de_indx2word = numericalize_vocab(nlp_de)
    with model.begin_training(train_X, train_Y, batch_size=nB, nb_epoch=nE) \
            as (trainer, optimizer):
        ''' add beginning and ending of sentence marks '''
        for pairs, pad_masks, lengths in trainer.batch_mask(train_X, train_Y):
            X_text, y_text = pairs
            X_mask, y_mask = pad_masks
            nX, nY = lengths
            ''' numericalize text '''
            X_num = []
            y_num = []
            for _x in X_text:
                X_num.append(numericalize(en_word2indx, _x))
            for _y in y_text:
                y_num.append(numericalize(de_word2indx, _y))
            ''' get text embeddings '''
            X_emb = en_embeddings(Model.ops.asarray(X_num))
            y_emb = de_embeddings(Model.ops.asarray(y_num))
            sentence_size = X_emb.shape[1]
            ''' add position encodings '''
            X = X_emb + X_positions[:sentence_size]
            y = y_emb + X_positions[:sentence_size]
            X = X.astype(Model.ops.xp.float32)
            y = y.astype(Model.ops.xp.float32)
            b = Batch((X, y), (X_mask, y_mask), (nX, nY))
            pdb.set_trace()
            # yh, backprop = model.begin_update(batch, drop=trainer.dropout)


if __name__ == '__main__':
    plac.call(main)
