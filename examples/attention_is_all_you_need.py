''' A driver file for attention is all you need paper demonstration '''
import plac
import spacy
from thinc.extra.datasets import get_iwslt
from thinc.loss import categorical_crossentropy
from spacy.lang.en import English
from spacy.lang.de import German
from thinc.neural.util import get_array_module
from spacy._ml import link_vectors_to_models
from thinc.neural.util import add_eos_bos, numericalize, numericalize_vocab, \
to_categorical
from thinc.neural._classes.encoder_decoder import EncoderDecoder
from thinc.neural._classes.embed import Embed
from thinc.v2v import Model
from timeit import default_timer as timer

MODEL_SIZE = 300
MAX_LENGTH = 50


def from_dataset(dev_X, dev_y, batch_size):
    ''' Get slice from input '''
    nX = Model.ops.xp.empty(batch_size)
    nY = Model.ops.xp.empty(batch_size)
    X_mask = Model.ops.xp.ones([batch_size, MAX_LENGTH],
                               dtype=Model.ops.xp.int)
    y_mask = Model.ops.xp.ones([batch_size, MAX_LENGTH],
                               dtype=Model.ops.xp.int)
    sent = 0
    for x_curr, y_curr in zip(dev_X, dev_y):
        x_pad = MAX_LENGTH - len(x_curr)
        y_pad = MAX_LENGTH - len(y_curr)
        if x_pad > 0:
            X_mask[sent][-x_pad:] = 0
        if y_pad > 0:
            y_mask[sent][-y_pad:] = 0
        nX[sent] = len(x_curr)
        nY[sent] = len(y_curr)
        x_curr.extend(['<pad>' for i in range(x_pad)])
        y_curr.extend(['<pad>' for i in range(y_pad)])
        sent += 1
    return (dev_X, dev_y), (X_mask, y_mask), (nX, nY)


def get_model_inputs(*args, dataset=False):
    ''' a function to prepare model inputs either directly from dataset
    or from a trainer slice '''
    if not dataset:
        nB, pairs, pad_masks, lengths, en_word2indx, de_word2indx, en_embeddings, \
        de_embeddings, X_positions = args
    else:
        nB, dev_X, dev_y, en_word2indx, de_word2indx, en_embeddings, \
        de_embeddings, X_positions = args
        pairs, pad_masks, lengths = from_dataset(dev_X, dev_y, nB)

    X_text, y_text = pairs
    X_mask, y_mask = pad_masks
    nL = X_mask.shape[1]
    nX, nY = lengths


    ''' numericalize text '''
    X_num = Model.ops.xp.empty([1, nB, nL], dtype=Model.ops.xp.int)
    y_num = Model.ops.xp.empty([1, nB, nL], dtype=Model.ops.xp.int)
    for i, _x in enumerate(X_text):
        X_num[0][i][:] = numericalize(en_word2indx, _x)
    for i, _y in enumerate(y_text):
        y_num[0][i][:] = numericalize(de_word2indx, _y)

    ''' get text embeddings '''
    X_emb0, backprop_X_emb0 = en_embeddings.begin_update(X_num)
    y_emb0, backprop_y_emb0 = de_embeddings.begin_update(y_num)

    ''' Text embeddings must be reshaped '''
    X_emb1 = X_emb0.reshape(nB, nL, MODEL_SIZE)
    y_emb1 = y_emb0.reshape(nB, nL, MODEL_SIZE)

    ''' add position encodings '''
    X = X_emb1 + X_positions[:nL]
    y = y_emb1 + X_positions[:nL]
    X = X.astype(Model.ops.xp.float32)
    y = y.astype(Model.ops.xp.float32)

    b0 = Batch((X, y), (X_mask, y_mask), (nX, nY))
    if not dataset:
        return b0, X, y, backprop_X_emb0, backprop_y_emb0, y_num
    else:
        return b0, y_num


class Batch:
    ''' Objects of this class share pairs X,Y along with their
        padding/future-words-hiding masks '''
    def __init__(self, pair, pad_masks, lengths):
        # X, y: nB x nL x nM
        # X_pad_mask, y_pad_mask: nB x nL
        # X_mask, y_mask: nB x nL x nL
        X, y = pair
        self.X_pad_mask, self.y_pad_mask = pad_masks
        nX, nY = lengths
        ''' TODO: this is super clumpsy and we need to fix this, but at least
        is happening only once ber batch, so it's acceptable for now '''
        self.y = y
        self.nB = X.shape[0]
        self.nL = X.shape[1]
        self.X = X
        self.X_mask = Model.ops.xp.repeat(self.X_pad_mask[:, :, Model.ops.xp.newaxis], self.nL, axis=2)
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
        # (nB, 1, nL) & (1, nL, nL) --> (nB, nL, nL)
        self.y_mask = Model.ops.xp.expand_dims(self.y_pad_mask, -2) & \
            self.subsequent_mask(self.nL)

    def subsequent_mask(self, nL):
        return (Model.ops.xp.triu(Model.ops.xp.ones([1, nL, nL]), k=1) == 0)


def spacy_tokenize(X_tokenizer, Y_tokenizer, X, Y, max_length=50):
    X_out = []
    Y_out = []
    for x, y in zip(X, Y):
        xdoc = X_tokenizer(x)
        ydoc = Y_tokenizer(y)
        if len(xdoc) < MAX_LENGTH and (len(ydoc) + 2) < MAX_LENGTH:
            X_out.append([w.text for w in xdoc])
            Y_out.append([w.text for w in ydoc])
    return X_out, Y_out


def resize_vectors(vectors):
    xp = get_array_module(vectors.data)
    shape = (int(vectors.shape[0]*1.1), vectors.shape[1])
    if not hasattr(xp, 'resize'):
        vectors.data = vectors.data.get()
        vectors.resize(shape)
        vectors.data = xp.array(vectors.data)
    else:
        vectors.resize(shape)


@plac.annotations(
    nH=("number of heads of the multiheaded attention", "option", "nH", int),
    dropout=("model dropout", "option"),
    nS=('Number of encoders/decoder in the enc/dec stack.', "option", "nS", int),
    nB=('Batch size for the training', "option", "nB", int),
    nE=('Number of epochs for the training', "option", "nE", int),
    use_gpu=("Which GPU to use. -1 for CPU", "option", "g", int),
    lim=("Number of sentences to load from dataset", "option", "l", int)
)
def main(nH=6, dropout=0.1, nS=6, nB=15, nE=20, use_gpu=-1, lim=2000):
    if use_gpu != -1:
        # TODO: Make specific to different devices, e.g. 1 vs 0
        spacy.require_gpu()
    train, dev, test = get_iwslt()
    train_X, train_Y = zip(*train)
    dev_X, dev_Y = zip(*dev)
    test_X, test_Y = zip(*test)
    X_positions = Model.ops.position_encode(MAX_LENGTH, MODEL_SIZE)
    print('Position encodings computed successfully')
    ''' Read dataset '''
    nlp_en = spacy.load('en_core_web_sm')
    nlp_de = spacy.load('de_core_news_sm')
    print('Models loaded')
    # eos_vector = Model.ops.xp.random.rand(MODEL_SIZE)
    # bos_vector = Model.ops.xp.random.rand(MODEL_SIZE)
    # resize_vectors(nlp_de.vocab.vectors)
    # nlp_de.vocab.set_vector('<eos>', eos_vector)
    # nlp_de.vocab.set_vector('<bos>', bos_vector)
    en_tokenizer = English().Defaults.create_tokenizer(nlp_en)
    de_tokenizer = German().Defaults.create_tokenizer(nlp_de)
    train_X, train_Y = spacy_tokenize(en_tokenizer, de_tokenizer,
                                      train_X[:lim],
                                      train_Y[:lim], MAX_LENGTH)
    dev_X, dev_Y = spacy_tokenize(en_tokenizer,
                                  de_tokenizer,
                                  dev_X[:lim], dev_Y[:lim], MAX_LENGTH)
    test_X, test_Y = spacy_tokenize(en_tokenizer, de_tokenizer,
                                    test_X[:lim], test_Y[:lim], MAX_LENGTH)
    train_Y = add_eos_bos(train_Y)
    # link_vectors_to_models(nlp_en.vocab)
    # link_vectors_to_models(nlp_de.vocab)
    # en_embeddings = StaticVectors(nlp_en.vocab.vectors.name, MODEL_SIZE, column=0)
    # de_embeddings = StaticVectors(nlp_de.vocab.vectors.name, MODEL_SIZE, column=0)
    en_word2indx, en_indx2word = numericalize_vocab(nlp_en)
    de_word2indx, de_indx2word = numericalize_vocab(nlp_de)
    ''' 1 for oov, 1 for pad '''
    nTGT = len(de_word2indx) + 2
    en_embeddings = Embed(MODEL_SIZE, nM=MODEL_SIZE, nV=nTGT)
    de_embeddings = Embed(MODEL_SIZE, nM=MODEL_SIZE, nV=nTGT)
    model = EncoderDecoder(nTGT=nTGT)



    def track_progress():
        b_dev, y = get_model_inputs(100, dev_X[:100], dev_Y[:100], en_word2indx,
                                        de_word2indx, en_embeddings,
                                        de_embeddings, X_positions, dataset=True)
        scores = model(b_dev)
        categorical_y = to_categorical(y, nb_classes=scores.shape[2])
        scores = scores.reshape(categorical_y.shape)
        if len(scores.shape) == 1:
            correct = ((scores >= 0.5) == (categorical_y >= 0.5)).sum()
        else:
            correct = (scores.argmax(axis=1) == categorical_y.argmax(axis=1)).sum()
        print(correct / categorical_y.shape[0])


    with model.begin_training(train_X, train_Y, batch_size=nB, nb_epoch=nE) \
            as (trainer, optimizer):
        trainer.dropout = dropout
        trainer.dropout_decay = 1e-4
        trainer.each_epoch.append(track_progress)
        for pairs, pad_masks, lengths in trainer.batch_mask(train_X, train_Y):
            nL = pad_masks[0].shape[1]
            b0, X, y, backprop_X_emb0, backprop_y_emb0, y_num = \
                get_model_inputs(nB, pairs, pad_masks, lengths, en_word2indx,
                                 de_word2indx, en_embeddings,
                                 de_embeddings, X_positions)
            yh, backprop = model.begin_update(b0, drop=trainer.dropout)
            yh2d = yh.reshape(-1, nTGT)
            y_num2d = y_num.reshape(-1)
            grad2d, loss = categorical_crossentropy(yh2d, y_num2d)
            grad = grad2d.reshape(nB, nL, nTGT)
            dX, dY = backprop(grad, optimizer)
            ''' the sum does not affect the grad '''
            dX_emb1, dy_emb1 = dX, dY
            dX_emb0 = dX_emb1.reshape(-1, MODEL_SIZE)
            dy_emb0 = dy_emb1.reshape(-1, MODEL_SIZE)
            backprop_X_emb0(dX_emb0, optimizer)
            backprop_y_emb0(dy_emb0, optimizer)


if __name__ == '__main__':
    plac.call(main)
