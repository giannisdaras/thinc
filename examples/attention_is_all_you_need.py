from thinc.neural.ops import NumpyOps, Ops
import plac
from thinc.extra.datasets import get_iwslt


@plac.annotations(
    heads=("number of heads of the multiheaded attention", "option"),
    dropout=("model dropout", "option")
)
def main(heads=6, dropout=0.1):
    train, dev, test = get_iwslt()
    X_train, Y_train = zip(*train)
    X_dev, Y_dev = zip(*dev)
    X_test, Y_test = zip(*test)


if __name__ == '__main__':
    plac.call(main)
