# coding: utf8
from .model import Model
from ...api import with_reshape
from .relu import ReLu
from .affine import Affine


class PositionwiseFeedForward(Model):
    def __init__(self, nM=300, nO=300, dropout=0.0):
        Model.__init__(self)
        self.ffd1 = with_reshape(ReLu(nM, nO))
        self.ffd2 = with_reshape(Affine(nO, nM))
        self.activation = ReLu()
        self.layers_ = [self.ffd1, self.ffd2]
        self.nO = nO

    def begin_update(self, X0, drop=0.1):
        X1, b_X1 = self.ffd1.begin_update(X0)
        ''' Use dropout only in activation '''
        X2, b_X2 = self.relu.begin_update(X1, drop=drop)
        X3, b_X3 = self.ffd2.begin_update(X2)

        def finish_update(dX3, sgd=None):
            dX2 = b_X3(dX3, sgd=sgd)
            dX1 = b_X2(dX2, sgd=sgd)
            dX0 = b_X1(dX1, sgd=sgd)
            return dX0
        return X3, finish_update
