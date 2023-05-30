from typing import Tuple, Union, Union

from pdll.autograd import Tensor

from ..functional import pool2d
from .module import Module


class Pool2d(Module):
    '''pooling
    '''
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding:Union[int, Tuple[int, ...]]=0, dilation: int=1, mode: str='max'):
        super().__init__()

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
        self.mode = mode

    def forward(self, data: Tensor) -> Tensor:
        return pool2d(data, self.kernel_size, self.stride, self.padding, self.dilation, self.mode)
        
    def ext_repr(self, ) -> str:
        return f'(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, mode={self.mode})'



class AvgPool2d(Pool2d):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding:Union[int, Tuple[int, ...]]=0, dilation: int=1):
        super().__init__(kernel_size, stride, padding, mode='avg')


class MaxPool2d(Pool2d):
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding:Union[int, Tuple[int, ...]]=0, dilation: int=1):
        super().__init__(kernel_size, stride, padding, mode='max')
