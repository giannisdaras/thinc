# coding: utf8
from .model import Model
from ...api import with_reshape
from .relu import ReLu
from .affine import Affine


class PositionwiseFeedForward(Model):
    def __init__(self, nM=300, nO=300, dropout=0.0):
        Model.__init__(self)
        self.ffd1 = with_reshape(ReLu(nI=nM, nO=nO))
        self.ffd2 = with_reshape(Affine(nI=nO, nO=nM))
        self.layers_ = [self.ffd1, self.ffd2]
        self.nO = nO

    def begin_update(self, X0, drop=0.0):
        X1, b_ffd1 = self.ffd1.begin_update(X0)
        X2, b_ffd2 = self.ffd2.begin_update(X1)

        def finish_update(dX2, sgd=None):
            dX1 = b_ffd2(dX2, sgd=sgd)
            dX0 = b_ffd1(dX1, sgd=sgd)
            return dX0
        return X2, finish_update
