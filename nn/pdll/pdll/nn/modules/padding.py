from typing import Union, Tuple

from pdll.autograd import Tensor

from ..functional import zero_pad2d, constant_pad2d
from .module import Module


__all__ = [
    'ZeroPad2d', 'ConstantPad2d', 
]

class ZeroPad2d(Module):
    '''ZeroPad2d
    '''
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]]):
        super().__init__()
        self.padding = padding

    def forward(self, data: Tensor) -> Tensor:
        return zero_pad2d(data, padding=self.padding)

    def ext_repr(self, ) -> str:
        return f'(padding={self.padding})'


class ConstantPad2d(Module):
    '''ConstantPad2d
    ''' 
    def __init__(self, padding: Union[int, Tuple[int, int, int, int]], value: float):
        super().__init__()
        self.padding = padding
        self.value = value
        
    def forward(self, data: Tensor) -> Tensor:
        return constant_pad2d(data, padding=self.padding, value=self.value)

    def ext_repr(self, ) -> str:
        return f'(padding={self.padding})'