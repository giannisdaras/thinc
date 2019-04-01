import pytest
import numpy as np
from thinc.v2v import Model
from ...neural._classes.encoder_decoder import EncoderDecoder, with_reshape, \
    MultiHeadedAttention
from ...neural._classes.affine import Affine
from ...neural.optimizers import SGD
from ...neural.ops import NumpyOps

class Batch:
    def __init__(self, pair, lengths):
        X, y = pair
        nX, nY = lengths
        self.X = X
        self.y = y
        self.nB = X.shape[0]
        self.nL = X.shape[1]
        self.X_mask = Model.ops.allocate((self.nB, self.nL, self.nL), dtype='bool')
        self.y_mask = Model.ops.allocate((self.nB, self.nL, self.nL), dtype='bool')
        for i, length in enumerate(nX):
            self.X_mask[i, :, :length] = 1
        for i, length in enumerate(nY):
            for j in range(length):
                self.y_mask[i, j, :j+1] = 1
            self.y_mask[i, length:, :length] = 1


@pytest.fixture
def model_properties():
    nM = 2
    nS = 1
    nH = 1
    return nM, nS, nH

@pytest.fixture
def sgd():
    return SGD(NumpyOps(), 0.001)

@pytest.fixture
def model(model_properties):
    nM, nS, nH = model_properties
    return EncoderDecoder(nS=nS, nH=nH, nM=nM)


@pytest.fixture
def beast(model_properties):
    nM, _, nH = model_properties
    return MultiHeadedAttention(nM=nM, nH=nH)


@pytest.fixture
def input_properties():
    nB = 3
    nL = 4
    length_X = np.array([2, 4, 3])
    length_y = np.array([2, 2, 3])
    return nB, nL, (length_X, length_y)


@pytest.fixture
def model_instances(input_properties, model_properties):
    nB, nL, lengths = input_properties
    nM, _, _ = model_properties
    X = np.random.rand(nB, nL, nM)
    y = np.random.rand(nB, nL, nM)
    return Batch((X, y), lengths)


def test_masks_shape(model_instances, input_properties):
    batch = model_instances
    nB, nL, _ = input_properties
    assert batch.X_mask.shape == (nB, nL, nL)
    assert batch.y_mask.shape == batch.X_mask.shape


def test_X_mask(model_instances, input_properties):
    batch = model_instances
    X_mask = batch.X_mask
    mask1 = np.array([[True, True, False, False],
                     [True, True, False, False],
                     [True, True, False, False],
                     [True, True, False, False]])
    mask2 = np.array([[True, True, True, True],
                     [True, True, True, True],
                     [True, True, True, True],
                     [True, True, True, True]])
    mask3 = np.array([[True, True, True, False],
                     [True, True, True, False],
                     [True, True, True, False],
                     [True, True, True, False]])
    correct_X_mask = np.array([mask1, mask2, mask3])
    assert np.array_equal(X_mask, correct_X_mask)


def test_y_mask(model_instances, input_properties):
    batch = model_instances
    y_mask = batch.y_mask
    mask1 = np.array([[True, False, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]])
    mask2 = np.array([[True, False, False, False],
                      [True, True, False, False],
                      [True, True, False, False],
                      [True, True, False, False]])
    mask3 = np.array([[True, False, False, False],
                      [True, True, False, False],
                      [True, True, True, False],
                      [True, True, True, False]])
    correct_y_mask = np.array([mask1, mask2, mask3])
    assert np.array_equal(y_mask, correct_y_mask)


def test_basic_with_reshape(sgd):
    X = np.random.rand(10, 20, 30)
    y = X
    model = with_reshape(Affine(30, 30))
    yh, backprop = model.begin_update(X)
    loss1 = ((yh - y) ** 2).sum()
    backprop(yh - y, sgd)
    yh, backprop = model.begin_update(X)
    loss2 = ((yh - y) ** 2).sum()
    assert loss2 < loss1

def test_attn_shapes(input_properties, model_properties, model_instances, beast):
    nB, nL, _ = input_properties
    nM, _, _ = model_properties
    model = beast
    batch = model_instances
    X = batch.X
    y = batch.y
    X_mask = batch.X_mask
    assert X.shape == (nB, nL, nM)
    assert y.shape == (nB, nL, nM)
    
