import numpy

try:
    from cupy import get_array_module
except ImportError:
    def get_array_module(*a, **k):
        return numpy


def categorical_crossentropy(scores, labels):
    xp = get_array_module(scores)
    target = xp.zeros(scores.shape, dtype='float32')
    loss = 0.
    for i in range(len(labels)):
        target[i, int(labels[i])] = 1.
        loss += (1.0-scores[i, int(labels[i])])**2
    return scores - target, loss


def L1_distance(vec1, vec2, labels, margin=0.2):
    xp = get_array_module(vec1)
    dist = xp.abs(vec1 - vec2).sum(axis=1)
    loss = (dist > margin) - labels
    return (sent1-sent2) * loss, (sent2-sent1) * loss, loss


def KLDIVLoss(scores_x, scores_y, reduced=False):
    ''' KLDIVLoss is better used after a label smoothing for the scores y.
    The formula is: l_i = y_i * ( logy_i - x_i)
    '''
    if reduced:
        raise NotImplementedError
    else:
        xp = get_array_module(scores_x)
        log_y = xp.log(scores_y)
        return xp.multiply(scores_y, log_y - scores_x)
