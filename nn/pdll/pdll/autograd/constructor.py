
from typing import Tuple, Union
from copy import deepcopy

from pdll.backend import executor

from .tensor import Tensor


__all__ = [
    'rand', 'randn', 'eye',
    'ones', 'ones_like',
    'zeros', 'zeros_like',
    'from_numpy',
]


# def tensor(data: Union[support_types]) -> Tensor:
#     return Tensor(deepcopy(data))


def rand(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.random.rand(*shape)
    return Tensor(data, requires_grad=requires_grad)


def randn(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.random.randn(*shape)
    return Tensor(data, requires_grad=requires_grad)


def zeros(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.zeros(shape)
    return Tensor(data, requires_grad=requires_grad)


def ones(*shape: Tuple[int], requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.ones(shape)
    return Tensor(data, requires_grad=requires_grad)


def zeros_like(t: Tensor, requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.zeros(t.shape, dtype=t.dtype)
    return Tensor(data, requires_grad=requires_grad)


def ones_like(t: Tensor, requires_grad=False) -> Tensor:
    '''
    '''
    data = executor.np.ones(t.shape, dtype=t.dtype)
    return Tensor(data, requires_grad=requires_grad)


def from_numpy(data: executor.support_types, requires_grad=False) -> Tensor:
    '''
    '''
    return Tensor(data, requires_grad=requires_grad)


def eye(*shape, requires_grad=False):
    '''
    '''
    data = executor.np.eye(shape)
    return Tensor(data, requires_grad=requires_grad)