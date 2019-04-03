from __future__ import unicode_literals
from .affine import Affine
from ... import describe
from ...describe import Synapses
from ...check import has_shape
from ... import check
from .model import Model
from ...describe import Dimension


@describe.attributes(
    nB=Dimension("Batch size"),
    nI=Dimension("Input size"),
)
class PureSoftmax(Model):
    ''' A pure softmax layer '''

    name = "pure softmax"

    @property
    def input_shape(self):
        return (self.nB, self.nI)

    @property
    def output_shape(self):
        return (self.nB, self.nI)

    def __init__(self, nI=None):
        Model.__init__(self)
        self.nI = nI

    @check.arg(1, has_shape(("nB", "nI")))
    def predict(self, input__BI):
        output__BO = self.ops.softmax(input__BI)
        return output__BO

    @check.arg(1, has_shape(("nB", "nI")))
    def begin_update(self, input__BI, drop=0.0):
        output__BO = self.predict(input__BI)

        @check.arg(0, has_shape(("nB", "nI")))
        def finish_update(grad__BO, sgd=None):
            grad__BI = grad__BO * output__BO
            sum_grad__BI = grad__BI.sum(axis=-1, keepdims=True)
            grad__BI -= sum_grad__BO * output__BO
            return grad__BI

        return output__BO, finish_update
