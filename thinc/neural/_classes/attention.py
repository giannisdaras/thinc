import math
from numpy import einsum
from ... import describe
from ...describe import Dimension, Synapses, Gradient
from .model import Model
from ..util import get_array_module


def _set_dimensions_if_needed(model, X, y=None):
    if model.nI is None:
        model.nI = X.shape[1]
    if model.nO is None and y is not None:
        if len(y.shape) == 2:
            model.nO = y.shape[1]
        else:
            model.nO = int(y.max()) + 1

@describe.attributes(
    nI=Dimension("Input width"),
    heads=Dimension("Number of heads for the multiheaded attention"),
    W=Synapses("Input weights",
        lambda obj: (obj.nK+obj.nK+obj.nO, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W)),
    d_W=Gradient("W"),
)
class MultiHeadedAttention(Model):
    ''' This class implements multiheaded attention. It can be used for self
    attention or outer attention, depending on our needs. There is no left
    and right context width. We attend to the whole sentence and we take
    care of the masks to adjust appropriately. There are no actual different
    weight matrices for each head, but a bigger weight matrix for all heads.
    Going to bigger dimensions is the key to get the multiple heads.
    For the time being; key, query and value matrices are supposed to have the
    same length.
    '''
    def __init__(self, heads=6, nI=None, **kwargs):
        super(MultiHeadedAttention, self).__init__(self, **kwargs)
        self.heads = heads
        self.nI = nI
        self.nK = nI // heads


    def attn(self, query, key, value, mask=None):
        ''' Compute attention based on query, key, value
        Expected data size: nB x nT x nK
        '''
        nB = query.shape[0]
        # number of tokens in each sentence
        nT = query.shape[1]
        # key length
        nK = query.shape[2]
        scores = self.ops.xp.matmul(query, key.transpose(0, 2, 1)) / math.sqrt(self.nI)
        # penalize masked tokens
        scores[self.ops.xp.where(mask == 0)] = 1e-9
        # TODO: fix this!
        # this could be done much faster if softmax was supported for >= 3d arrays
        for batch in range(nB):
            scores[batch, :, :] = self.ops.softmax(scores[batch, :, :])
        ''' Now the dimensions of scores are:
        nB x nT x nT
        We multiply by values which is nB x nT x nK, so the result is:
        nB x nT x nK
        '''
        real_scores = self.ops.xp.matmul(scores, value)
        return real_scores







@describe.attributes(
    nK=Dimension("Key width"),
    nO=Dimension("Values width"),
    nI=Dimension("Input width"),
    nL=Dimension("Left context width"),
    nR=Dimension("Right context width"),
    W=Synapses("Input weights",
        lambda obj: (obj.nK+obj.nK+obj.nO, obj.nI),
        lambda W, ops: ops.xavier_uniform_init(W)),
    d_W=Gradient("W"),
)
class SelfAttention(Model):
    def __init__(self, nK=None, nO=None, nI=None, nL=5, nR=5, **kwargs):
        Model.__init__(self, **kwargs)
        self.nK = nK
        self.nO = nO
        self.nL = nL
        self.nR = nR
        self.nI = nI

    def begin_update(self, X_lengths, drop=0.):
        X, lengths = X_lengths

        (queries, keys, values), get_dX = self.project_inputs(X, lengths)
        attention, backprop_compare = self.compare(queries, keys, lengths)
        output, backprop_rescale = self.rescale(values, attention, lengths,
                                                self.nL, self.nR)

        def self_attention_finish_update(d_output, sgd=None):
            d_values, d_attention = backprop_rescale(d_output)
            d_queries, d_keys = backprop_compare(d_attention)
            dX = get_dX(d_queries, d_keys, d_values)
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dX

        return (output, lengths), self_attention_finish_update

    def project_inputs(self, X, lengths):
        # Let's say X is (25, 300)
        # If nK=32 and nO=64, we need to project down to 32+32+64=128
        # So we have a weight matrix of shape (300, 128)
        Y = self.ops.gemm(X, self.W, trans2=True) # (25, 128)
        queries = Y[:, :self.nK] # Shape (25, 32)
        keys = Y[:, self.nK:self.nK*2]
        # This will be shape (sum(kv_lengths), 64)
        values = Y[:, self.nK*2:]

        def backprop_get_inputs(d_queries, d_keys, d_values):
            dY = self.ops.xp.hstack((d_queries, d_keys, d_values)) # (25, 128)
            # ab,cb->ac
            dX = self.ops.gemm(dY, self.W)
            # ac,ab->cb
            self.ops.gemm(dY, X, out=self.d_W, trans1=True)
            return dX

        return (queries, keys, values), backprop_get_inputs

    def compare(self, queries, keys, lengths):
        '''Compare queries and keys according to scaled dot product attention.

        Queries and keys should be equally-shaped ragged arrays, representing
        variable length sequences. Typically each row will represent a word.

        We return a ragged matrix which if padded would be of shape:
        (sum(lengths), window_size)
        '''
        (dotprod, dotprod_lengths), backprop_rwd = _ragged_window_dot(
            self.ops, queries, keys, lengths, self.nL, self.nR)
        dotprod /= self.ops.xp.sqrt(self.nK)
        attention = softmax_ragged(self.ops, dotprod, dotprod_lengths)

        def backprop_attention(d_attention):
            d_dotprod = backprop_softmax_ragged(self.ops, d_attention, attention,
                                                dotprod_lengths)
            d_dotprod /= self.ops.xp.sqrt(self.nO)
            d_queries, d_keys = backprop_rwd(d_dotprod)
            return d_queries, d_keys

        return attention, backprop_attention

    def rescale(self, V, A, lengths, nL, nR):
        '''Perform a weighted sum of values with the attention.

        Values is a ragged array of sequences, unpacked it would be
        [(N1, d), (N2, d), ...etc], where N1 is the length of sequence 1,
        N2 is the length of sequence 2, etc.

        Attention is a ragged array of rows, where each row is a word's
        attention weights, and the weights are of varying length at the edge
        of each sequence.
        '''
        output = self.ops.allocate(V.shape)
        seq_idx = 0
        word_idx = 0
        att_idx = 0
        for i, length in enumerate(lengths):
            V_ = V[seq_idx : seq_idx + length]
            for j in range(length):
                values = get_window(V_, j, nL, nR)
                #values = V_[max(0, j-nL) : j+nR+1]
                attention = A[att_idx : att_idx + values.shape[0]]
                attention = attention.reshape((attention.size, 1))
                # set row of d from ((w, d) * (w, d)).sum()
                output[word_idx] = (values * attention).sum(axis=0)
                word_idx += 1
                att_idx += values.shape[0]
            seq_idx += length

        V_shape = tuple(V.shape)
        A_shape = tuple(A.shape)
        def backprop_rescale(d_output):
            dV = self.ops.allocate(V_shape)
            dA = self.ops.allocate(A_shape)
            seq_idx = 0
            word_idx = 0
            att_idx = 0
            for i, length in enumerate(lengths):
                V_ = V[seq_idx : seq_idx + length]
                dV_ = dV[seq_idx : seq_idx + length]
                for j in range(length):
                    d_out_word = d_output[word_idx]
                    #values    = V_[max(0, j-nL) : j+nR+1]
                    values = get_window(V_, j, nL, nR)
                    attention = A[att_idx : att_idx + values.shape[0]]

                    #dV_[max(0, j-nL) : j+nR+1] += self.ops.xp.outer(attention, d_out_word)
                    inc_window(dV_, j, nL, nR, self.ops.xp.outer(attention, d_out_word))
                    dA[att_idx : att_idx + values.shape[0]] += (values * d_out_word).sum(axis=-1)
                    word_idx += 1
                    att_idx += values.shape[0]
                seq_idx += length
            return dV, dA
        return output, backprop_rescale

def get_window(X, i, nL, nR):
    xp = get_array_module(X)
    lefts = X[max(i-nL, 0) : i]
    rights = X[i : i+nR+1]
    return xp.vstack((lefts, rights))

def inc_window(X, i, nL, nR, Y):
    left_start = max(i-nL, 0)
    right_end = i+nR+1
    X[left_start : i] += Y[:i-left_start]
    X[i : right_end] += Y[i-left_start:]

def softmax_ragged(ops, X, lengths):
    Y = ops.allocate(X.shape)
    start = 0
    for i, length in enumerate(lengths):
        Y[start : start+length] = ops.softmax(X[start:start+length])
        start += length
    return Y


def backprop_softmax_ragged(ops, dY, Y, lengths):
    dX = dY * Y
    start = 0
    for i, length in enumerate(lengths):
        sumdx = dX[start : start+length].sum()
        dX[start : start+length] -= Y[start : start+length] * sumdx
        start += length
    return dX


def _ragged_window_dot(ops, X, Y, lengths, nL, nR):
    '''Multiply X against context windows of Y, where X and Y are both ragged
    matrices, representing concatenated sequences. We output a ragged array
    where each entry is a vector with the dot product of X[i] against the
    context-window of Y, where the context-window is of variable length'''
    # Imagine we had the simple thing (without the window, no raggedness etc):
    #
    # output = einsum('nd,nd->nn', X, Y)
    # dX     = einsum('nn,nd->nd', d_output, Y)
    # dY     = einsum('nn,nd->nd', d_output, X)
    #
    # Okay. Now let's say we expand and pad, wrapping that in an op:
    #
    # winY, backprop_window = window(Y, window_size)
    #
    # where winY is of shape (n, w, d) for w = window_size*2+1
    #
    # Then:
    #
    # output = einsum('nd,nwd->nw', X, winY)
    # dX     = einsum('nw,nwd->nd', d_output, win_Y)
    # d_winY = einsum('nw,nd->nwd', d_output, X)
    # dY = backprop_window(d_winY)
    start = 0
    output = []
    backprops = []
    for i, length in enumerate(lengths):
        X_ = X[start : start+length]
        Y_ = Y[start : start+length]
        for j in range(length):
            dots, backprop = _window_dot(X_, Y_, j, nL, nR)
            output.append(dots)
            backprops.append(backprop)
        start += length
    out_lengths = ops.asarray([len(out) for out in output])
    shape = tuple(X.shape)

    def backprop_rwd(d_output):
        d_output_list = ops.unflatten(d_output, out_lengths)
        dX = ops.allocate(shape)
        dY = ops.allocate(shape)
        for i, (d_dots, backprop) in enumerate(zip(d_output_list, backprops)):
            dX[i], dY[i] = backprop(d_dots)
        return dX, dY

    # We do have to output a ragged array here...If we pad, it'll be frustrating
    # later, because our softmax will be off due to the padding.
    return (ops.flatten(output), out_lengths), backprop_rwd


def _window_dot(X, Y, i, nL, nR):
    start = max(0, i-nL)
    end = i + nR + 1
    output = einsum('d,wd->w', X[i], Y[start : end])

    def backprop_window_dot(d_output):
        dXi = einsum('w,wd->d', d_output, Y[start : end])
        d_winY = einsum('w,d->wd', d_output, X[i])
        return dXi, d_winY.sum(axis=0)

    return output, backprop_window_dot


@describe.attributes(
    nO=Dimension("Output size"),
    Q=Synapses("Learned 'query' vector",
        lambda obj: (obj.nO, 1),
        lambda Q, ops: ops.normal_init(Q, Q.shape[0])),
    dQ=Gradient("Q"),
)
class ParametricAttention(Model):
    """Weight inputs by similarity to a learned vector"""
    name = 'para-attn'

    def __init__(self, nO=None, hard=False, **kwargs):
        Model.__init__(self, **kwargs)
        self.nO = nO
        self.hard = hard
        self.drop_factor = kwargs.get('drop_factor', 1.0)

    def begin_update(self, Xs_lengths, drop=0.):
        Xs, lengths = Xs_lengths
        attention, bp_attention = self._get_attention(self.Q, Xs, lengths)
        output, bp_output = self._apply_attention(attention, Xs, lengths)

        def attention_bwd(d_output, sgd=None):
            dXs, d_attention = bp_output(d_output)
            dQ, dXs2 = bp_attention(d_attention)
            self.dQ += dQ
            dXs += dXs2
            if sgd is not None:
                sgd(self._mem.weights, self._mem.gradient, key=self.id)
            return dXs
        return (output, lengths), attention_bwd

    def _get_attention(self, Q, Xs, lengths):
        attention = Xs.dot(Q)
        if self.hard:
            start = 0
            for i, length in enumerate(lengths):
                argmax = attention[start:start+length].argmax()
                attention[start:start+length] = 0
                attention[start+argmax] = 1.
                start += length
        else:
            attention = self.ops.softmax_sequences(attention, lengths)
        def get_attention_bwd(d_attention):
            if self.hard:
                d_attention *= attention
            else:
                d_attention = self.ops.backprop_softmax_sequences(d_attention,
                                                                 attention,
                                                                 lengths)
            dQ = self.ops.gemm(Xs, d_attention, trans1=True)
            dXs = self.ops.xp.outer(d_attention, Q)
            return dQ, dXs
        return attention, get_attention_bwd

    def _apply_attention(self, attention, Xs, lengths):
        output = Xs * attention
        def apply_attention_bwd(d_output):
            d_attention = (Xs * d_output).sum(axis=1, keepdims=True)
            dXs         = d_output * attention
            return dXs, d_attention
        return output, apply_attention_bwd
