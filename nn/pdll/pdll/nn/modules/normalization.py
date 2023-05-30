
from pdll.autograd import Tensor

from ..initialization import ones, zeros
from ..parameter import Parameter
from .module import Module


class BatchNorm2d(Module):
    '''bn
    https://arxiv.org/abs/1502.03167

    from autograd import variable
    import torch
    import numpy as np

    data = executor.np.random.rand(3, 8, 10, 10).astype(executor.np.float32)
    bn = torch.nn.BatchNorm2d(8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    bn_var = variable.BatchNorm2d(8, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, training=True)

    t = torch.tensor(data, requires_grad=True)
    out_t = bn(t)
    out_t.mean().backward()

    print('-------------------------')

    v = variable.Tensor(data, requires_grad=True)
    out_v = bn_var(v)
    out_v.mean().backward()

    executor.np.testing.assert_almost_equal(out_t.data.numpy(), out_v.data, decimal=4)
    executor.np.testing.assert_almost_equal(bn.running_mean.data.numpy(), bn_var.running_mean, decimal=4)
    executor.np.testing.assert_almost_equal(bn.running_var.data.numpy(), bn_var.running_var, decimal=4)
    executor.np.testing.assert_almost_equal(bn.weight.data.numpy(), bn_var.weight.data, decimal=4)
    executor.np.testing.assert_almost_equal(bn.bias.data.numpy(), bn_var.bias.data, decimal=4)
    executor.np.testing.assert_almost_equal(t.grad.data.numpy(), v.grad, decimal=4)

    '''

    def __init__(self, num_features: int, momentum: float=0.1, eps: float=1e-05, affine: bool=True, track_running_stats: bool=True, training=True):
        super().__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training

        self.weight = Parameter(data=ones(shape=(num_features, )))
        self.bias = Parameter(data=zeros(shape=(num_features, )))

        if not self.affine:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

        self.running_mean = None
        self.running_var = None

        if self.track_running_stats:
            self.running_mean = Tensor(zeros((num_features, ))) # N, H, W
            self.running_var = Tensor(ones((num_features)))
            self.running_num_batches = 0
            self.register_buffer('running_mean', self.running_mean)
            
    def ext_repr(self, ) -> str:
        return f'(num_features={self.num_features}, training={self.training}, momentum={self.momentum}, affine={self.affine})'

    def forward(self, data: Tensor) -> Tensor:
        if self.training:
            mean = data.mean(axis=(0, 2, 3), keepdims=True)
            # var = ((data - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            var = data.var(axis=(0, 2, 3), keepdims=True)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)

        out = (data - mean) / (var.sqrt() + self.eps) # .reshape(1, -1, 1, 1)

        if self.affine:
            out = self.weight.reshape(1, self.num_features, 1, 1) * out + self.bias.reshape(1, self.num_features, 1, 1)

        if self.track_running_stats and self.training:
            self.running_mean = self.running_mean * (1 - self.momentum) + mean.data[0, :, 0, 0] * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + var.data[0, :, 0, 0] * self.momentum
            self.running_num_batches += 1

        return out





class GroupNorm2d(Module):

    def __init__(self, num_features, num_groups, eps=1e-6, use_bias=True):
        super().__init__()

        self.num_groups = num_groups
        self.num_features = num_features
        self.weight = Parameter(ones((1, num_features, 1, 1)))
        if use_bias:
            self.bias = Parameter(zeros((1, num_features, 1, 1)))
        self.eps = eps
        self.use_bias = use_bias

    def forward(self, data: Tensor) -> Tensor:

        n, c, h, w = data.shape
        assert c == self.num_features, ''
        assert c % self.num_groups == 0, ''

        data = data.reshape(n, self.num_groups, -1)

        mean = data.mean(axis=-1, keepdims=True)
        var = data.var(axis=-1, keepdims=True)

        data = (data - mean) / (var.sqrt() + self.eps)

        if self.use_bias:
            out = data * self.weight + self.bias
        else:
            out = data * self.weight

        return out.reshape(n, c, h, w)