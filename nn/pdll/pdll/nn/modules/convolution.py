import math
from typing import Union, Tuple

from pdll.autograd import Tensor

from ..parameter import Parameter
from ..functional import conv2d
from ..initialization import uniform
from .module import Module


class Conv2d(Module):
    '''
    image: C_in H_in W_in
    kernel: C_out C_in H_kernel W_kernel
    output: C_out H_out W_out
    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    '''
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding: Union[int, Tuple[int, ...]]=0, dilation: int=1, groups: int=1, bias: bool=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, (tuple, list)) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])
        elif isinstance(padding, (tuple, list)) and len(padding) == 4:
            padding = tuple(padding)
        else:
            raise RuntimeError('not suppot padding format')

        self.padding = padding

        self.dilation = dilation
        self.groups = groups
        
        assert in_channels % groups == 0, 'k * 1/k * 1/k'

        k = math.sqrt(1. / (groups * in_channels * kernel_size[0] * kernel_size[1]))
        weight_init = uniform(low=-k, high=k, size=(out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1]))
        self.weight = Parameter(data=weight_init)
        
        if bias:
            bias_init = uniform(low=-k, high=k, size=(self.out_channels, ))
            self.bias = Parameter(data=bias_init)
        else:
            self.bias = None

    def forward(self, data: Tensor) -> Tensor:
        return conv2d(data, self.weight, self.bias, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)

    def ext_repr(self, ) -> str:
        s = f'({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'
        if self.dilation != 1:
            s += f'dilation={self.dilation}'
        if self.groups != 1:
            s += f'groups={self.groups}'

        return s
