from thinc.neural.ops import NumpyOps, Ops
import plac


@plac.annotations(
    heads=("number of heads of the multiheaded attention", "positional",
           None, int),
    dropout=("model dropout", "option")
)
def main(heads, dropout=0.1):
    print(dropout)


if __name__ == '__main__':
    plac.call(main)
