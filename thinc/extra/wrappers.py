# coding: utf8
from __future__ import unicode_literals

import contextlib
from ..compat import BytesIO
from ..neural._classes.model import Model

try:
    import cupy
except ImportError:
    cupy = None

try:
    import torch.autograd
    import torch.optim
    import torch
    import torch.utils.dlpack
except ImportError:
    torch = None


def xp2torch(xp_tensor):
    if hasattr(xp_tensor, "toDlpack"):
        return torch.utils.dlpack.from_dlpack(xp_tensor.toDlpack())
    else:
        return torch.from_numpy(xp_tensor)


def torch2xp(torch_tensor):
    if torch_tensor.is_cuda:
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(torch_tensor))
    else:
        return torch_tensor.detach().numpy()


class PyTorchWrapper(Model):
    """Wrap a PyTorch model, so that it has the same API as Thinc models.
    To optimize the model, you'll need to create a PyTorch optimizer and call
    optimizer.step() after each batch --- see examples/wrap_pytorch.py
    """

    def __init__(self, model, conf=None):
        Model.__init__(self)
        self._model = model
        self._optimizer = None
        self.conf = conf

    def begin_update(self, x_data, drop=0.0):
        """Return the output of the wrapped PyTorch model for the given input,
        along with a callback to handle the backward pass.
        """
        if self.conf is None:
            x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=True)
            y_var = self._model(x_var)
            def backward_pytorch(dy_data, sgd=None):
                dy_var = xp2torch(dy_data)
                torch.autograd.backward((y_var,), grad_tensors=(dy_var,))
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                return torch2xp(x_var.grad)
            y = torch2xp(y_var)
        else:
            '''
                self.grads specifies how the pytorch wrapper should behave
                when multiple inputs come into play.
                Args:
                    i_grad: optimizable input parameters (bool)
                    b_grad: backpropagatable output parameters (bool)
                    i_no_grad: return input arguments with no grad (bool)
                    o_xp: output parameters to tensors (bool)
            '''
            i_grad, b_grad, i_no_grad, o_xp = self.conf
            x_var = [x_data[i] if not grad else \
                    torch.autograd.Variable(xp2torch(x_data[i]), \
                                            requires_grad=True) for i, grad \
                                            in enumerate(i_grad)]
            y_var = self._model(x_var)
            y = []
            for i, flag in enumerate(o_xp):
                if flag:
                    y.append(torch2xp(y_var[i]))
                else:
                    y.append(y_var[i])
            y = tuple(y)
            def backward_pytorch(dy_data, sgd=None):
                dy_var = xp2torch(dy_data)
                for i, data in enumerate(y_var):
                    if b_grad[i]:
                        torch.autograd.backward((y_var[i],), grad_tensors=(dy_var,))
                if sgd is not None:
                    if self._optimizer is None:
                        self._optimizer = self._create_optimizer(sgd)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                if i_no_grad is not None:
                    b_out = []
                    for i, ret in enumerate(i_no_grad):
                        if ret and i_grad[i]:
                            b_out.append(torch2xp(x_var[i].grad))
                        elif ret:
                            b_out.append(torch2xp(x_var[i]))
                    return tuple(b_out)
                else:
                    return torch2xp(x_var[0].grad)
        return y, backward_pytorch


    def _create_optimizer(self, sgd):
        params = self._model.parameters()
        if sgd.b1 != 0 and sgd.b2 != 0:
            optimizer = torch.optim.Adam(params, lr=sgd.alpha, betas=(sgd.b1, sgd.b2))
        elif sgd.b2 == 0:
            optimizer = torch.optim.SGD(params, lr=sgd.alpha, momentum=sgd.b1)
        else:
            raise NotImplementedError
        return optimizer

    def to_disk(self, path):
        # TODO: Untested
        torch.save(self._model.state_dict(), str(path))

    def from_disk(self, path):
        # TODO: Untested
        self._model.load_state_dict(torch.load(path))

    def to_bytes(self):
        # TODO: Untested
        filelike = BytesIO()
        torch.save(self._model.state_dict(), filelike)
        filelike.seek(0)
        return filelike.getvalue()

    def from_bytes(self, data):
        # TODO: Untested
        filelike = BytesIO(data)
        filelike.seek(0)
        self._model.load_state_dict(torch.load(filelike))

    def to_gpu(self, device_num):
        self._model.cuda(device_num)

    def to_cpu(self):
        self._model.cpu()

    def resize_output(self, new_dim):
        # self.weight = nn.Parameter(F.pad(self.weight, ...)) # add classes
        # self.weight = nn.Parameter(F.pad(model.weight, ...)) # add classes
        raise NotImplementedError

    def resize_input(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        if self.id in params:
            backup = self.to_bytes()
            self.from_bytes(params[self.id])
        else:
            backup = None
        yield
        if backup is not None:
            self.from_bytes(backup)


class PyTorchWrapperRNN(PyTorchWrapper):
    """Wrap a PyTorch RNN model
    """

    def __call__(self, x_data, h_0=None):
        x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=False)
        # Make prediction
        out, h_n = self._model(x_var, h_0)
        return (self.ops.asarray(out.data), h_n)

    def begin_update(self, x_data, h_0=None, drop=0.0):
        """Return the output of the wrapped PyTorch model for the given input,
        along with a callback to handle the backward pass.
        """
        x_var = torch.autograd.Variable(xp2torch(x_data), requires_grad=True)

        # Make prediction
        out, h_n = self._model(x_var, h_0)
        # Shapes will be:
        # out = seq_len, batch, hidden_size * num_directions
        # h_n = num_layers * num_directions, batch, hidden_size

        def backward_pytorch_rnn(d_data, sgd=None):
            dy_data, _ = d_data
            dout = xp2torch(dy_data)
            torch.autograd.backward((out,), grad_tensors=(dout,))
            if sgd is not None:
                if self._optimizer is None:
                    self._optimizer = self._create_optimizer(sgd)
                self._optimizer.step()
                self._optimizer.zero_grad()
            return torch2xp(x_var.grad)

        return (torch2xp(out), h_n), backward_pytorch_rnn

    def resize_output(self, new_dim):
        # self.weight = nn.Parameter(F.pad(self.weight, ...)) # add classes
        # self.weight = nn.Parameter(F.pad(model.weight, ...)) # add classes
        raise NotImplementedError

    def resize_input(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        if self.id in params:
            backup = self.to_bytes()
            self.from_bytes(params[self.id])
        else:
            backup = None
        yield
        if backup is not None:
            self.from_bytes(backup)
