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
