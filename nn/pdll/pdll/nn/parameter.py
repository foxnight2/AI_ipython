from typing import Tuple, Union

from pdll.autograd import Tensor

from .initialization import uniform, zeros

class Parameter(Tensor):
    '''parameter
    '''
    def __init__(self, *shape, data=None):
        if shape:
            data = uniform(-1, 1, size=shape)
        super().__init__(data, requires_grad=True)
        self.grad = Tensor(zeros(self.shape))
