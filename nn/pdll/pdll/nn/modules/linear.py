import math

from pdll.autograd import Tensor

from ..parameter import Parameter
from ..initialization import uniform

from .module import Module

class Linear(Module):
    """Linear 
    """
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        k = math.sqrt(1. / input_dim)
        init_weight = uniform(low=-k, high=k, size=(input_dim, output_dim))
        self.weight = Parameter(data=init_weight)
        if self.use_bias:
            init_bias = uniform(low=-k, high=k, size=(output_dim, ))
            self.bias = Parameter(data=init_bias)
        
    def forward(self, data: Tensor) -> Tensor:
        if self.use_bias:
            return data @ self.weight + self.bias
        else:
            return data @ self.weight

    def ext_repr(self, ) -> str:
        return f'({self.input_dim}, {self.output_dim})'    
