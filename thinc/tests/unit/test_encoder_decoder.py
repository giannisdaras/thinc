import pytest
import numpy as np
from thinc.v2v import Model
from ...neural._classes.encoder_decoder import EncoderDecoder

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
