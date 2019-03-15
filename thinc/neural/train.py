# coding: utf8
from __future__ import unicode_literals

import numpy.random
from tqdm import tqdm

from .optimizers import Adam, linear_decay


class Trainer(object):
    def __init__(self, model, **cfg):
        self.ops = model.ops
        self.model = model
        self.L2 = cfg.get("L2", 0.0)
        self.optimizer = Adam(model.ops, 0.001, decay=0.0, eps=1e-8, L2=self.L2)
        self.batch_size = cfg.get("batch_size", 128)
        self.nb_epoch = cfg.get("nb_epoch", 20)
        self.i = 0
        self.dropout = cfg.get("dropout", 0.0)
        self.dropout_decay = cfg.get("dropout_decay", 0.0)
        self.each_epoch = []

    def __enter__(self):
        return self, self.optimizer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.use_params(self.optimizer.averages)

    def iterate(self, train_X, train_y, progress_bar=True):
        orig_dropout = self.dropout
        for i in range(self.nb_epoch):
            indices = numpy.arange(len(train_X))
            numpy.random.shuffle(indices)
            indices = self.ops.asarray(indices)
            j = 0
            with tqdm(total=indices.shape[0], leave=False) as pbar:
                while j < indices.shape[0]:
                    slice_ = indices[j : j + self.batch_size]
                    X = _take_slice(train_X, slice_)
                    y = _take_slice(train_y, slice_)
                    yield X, y
                    self.dropout = linear_decay(orig_dropout, self.dropout_decay, j)
                    j += self.batch_size

                    if progress_bar:
                        pbar.update(self.batch_size)
            for func in self.each_epoch:
                func()

    def batch_mask(self, train_X, train_y, progress_bar=True, pad_token=0):
        for i in range(self.nb_epoch):
            indices = self.ops.xp.arange(len(train_X))
            self.ops.xp.random.shuffle(indices)
            indices = self.ops.asarray(indices)
            j = 0
            with tqdm(total=indices.shape[0], leave=False) as pbar:
                slice_ = indices[j : j + self.batch_size]
                X = _take_slice(train_X, slice_)
                y = _take_slice(train_y, slice_)
                j += self.batch_size
                max_sent = 0
                for i, j in zip(X, y):
                    curr_len = max(len(i), len(j))
                    if (curr_len > max_sent):
                        max_sent = curr_len
                nX = self.ops.xp.empty(self.batch_size)
                nY = self.ops.xp.empty(self.batch_size)
                X_mask = self.ops.xp.ones([self.batch_size, max_sent], dtype=self.ops.xp.int)
                y_mask = self.ops.xp.ones([self.batch_size, max_sent], dtype=self.ops.xp.int)
                sent = 0
                for x_curr, y_curr in zip(X, y):
                    x_pad = max_sent - len(x_curr)
                    y_pad = max_sent - len(y_curr)
                    ''' this if is a bit ugly, but slicing gets really
                    weird if you end up with a zero here '''
                    if x_pad > 0:
                        X_mask[sent][-x_pad:] = 0
                    if y_pad > 0:
                        y_mask[sent][-y_pad:] = 0
                    nX[sent] = len(x_curr)
                    nY[sent] = len(y_curr)
                    x_curr.extend(['<pad>' for i in range(x_pad)])
                    y_curr.extend(['<pad>' for i in range(y_pad)])
                    sent += 1
                yield (X, y), (X_mask, y_mask), (nX, nY)
                if progress_bar:
                    pbar.update(self.batch_size)
            for func in self.each_epoch:
                func()



def _take_slice(data, slice_):
    if isinstance(data, list) or isinstance(data, tuple):
        return [data[int(i)] for i in slice_]
    else:
        return data[slice_]
