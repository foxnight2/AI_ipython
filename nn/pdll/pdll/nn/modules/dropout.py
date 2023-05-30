from typing import Tuple, Optional

from pdll.autograd import Tensor

from ..functional import dropout
from .module import Module



class Dropout(Module):
    '''dropout
    '''
    def __init__(self, p: float, training: bool=True, inspace: bool=True):
        super().__init__()
        self.p = p 
        self.training = training
        self.inspace = inspace

    def forward(self, data: Tensor) -> Tensor:
        return dropout(data, self.p, self.training, self.inspace)

    def ext_repr(self, ) -> str:
        return f'(p={self.p}, training={self.training})'
