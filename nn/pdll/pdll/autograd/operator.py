from typing import Tuple, Iterable, Iterator, Union

from functools import reduce as REDUCE
from operator import mul as MUL

from pdll.backend import executor

from .function import Function
from .tensor import Tensor
from .backpropag import Leaf

from .utils import broadcast_reverse
from .utils import register
from .utils import to_tensor


__all__ = [
    'add', 'sub', 'mul', 'neg', 'div', 'matmul', 'pow', 'exp',
    'sum', 'mean', 'var', 
    'reshape', 'transpose', 'flip',
]

class _Add(Function):
    '''a + b
    add broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    '''
    def forward(self, a: Union[executor.support_types], b: Union[executor.support_types]) -> Union[executor.support_types]:
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad: Union[executor.support_types]) -> Tuple[Union[executor.support_types], Union[executor.support_types]]:
        assert self.c_shape == grad.shape, 'add' 
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)
        return a_grad, b_grad


class _Sub(Function):
    '''a - b
    '''
    def forward(self, a: Union[executor.support_types], b: Union[executor.support_types]) -> Union[executor.support_types]:
        c = a - b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad):
        assert grad.shape == self.c_shape, 'sub'
        a_grad = broadcast_reverse( grad, self.a_shape)
        b_grad = broadcast_reverse(-grad, self.b_shape)
        return a_grad, b_grad 


class _Neg(Function):
    '''-t 
    '''
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        return -t 
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        return -grad


class _Mul(Function):
    '''a * b
    '''
    def forward(self, a: Union[executor.support_types], b: Union[executor.support_types]) -> Union[executor.support_types]:
        c = a * b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        return c
    
    def backward(self, grad):
        assert self.c_shape == grad.shape, 'mul'
        a_grad = broadcast_reverse(grad * self.b, self.a.shape)
        b_grad = broadcast_reverse(grad * self.a, self.b.shape)
        return a_grad, b_grad


class _Div(Function):
    '''a / b
    '''
    def forward(self, a: Union[executor.support_types], b: Union[executor.support_types], eps: float=1e-10) -> Union[executor.support_types]:
        # executor.np.testing.assert_almost_equal(b, 0)
        c = a / b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        self.eps = eps
        return c
    
    def backward(self, grad: Union[executor.support_types]):
        assert grad.shape == self.c_shape
        a_grad = grad / self.b
        b_grad = -grad * self.a / (self.b ** 2 + 1e-10)
        a_grad = broadcast_reverse(a_grad, self.a.shape)
        b_grad = broadcast_reverse(b_grad, self.b.shape)
        return a_grad, b_grad


class _Matmul(Function):
    '''t1 @ t2
    t1 @ t2 [2, 3] [3, 5] -> [2, 5]
    grad @ t2.T [2, 5] [5, 3] -> [2, 3]
    t1.T @ grad [3, 2] [2, 5] -> [3, 5]
    '''
    def forward(self, t1: Union[executor.support_types], t2: Union[executor.support_types]) -> Union[executor.support_types]:
        # assert t1.ndim == t1.ndim, ''
        out = t1 @ t2
        self.t1 = t1
        self.t2 = t2
        self.out_shape = out.shape
        return out
    
    def backward(self, grad: Union[executor.support_types]) -> Tuple[Union[executor.support_types]]:
        assert grad.shape == self.out_shape, ''

        t1_shape = self.t1.shape
        t2_shape = self.t2.shape
        t1_shape_t_idx = list(range(len(t1_shape)-2)) + list(range(len(t1_shape)-2, len(t1_shape)))[::-1]
        t2_shape_t_idx = list(range(len(t2_shape)-2)) + list(range(len(t2_shape)-2, len(t2_shape)))[::-1]
        
        grad_t1 = grad @ self.t2.transpose(t2_shape_t_idx)
        grad_t2 = self.t1.transpose(t1_shape_t_idx) @ grad

        grad_t1 = broadcast_reverse(grad_t1, self.t1.shape)
        grad_t2 = broadcast_reverse(grad_t2, self.t2.shape)

        return grad_t1, grad_t2


class _GetItem(Function):
    '''getitem
    '''
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    def forward(self, t: Union[executor.support_types]):
        self.t_shape = t.shape
        return t[self.index]
    
    def backward(self, grad):
        _grad = executor.np.zeros(shape=self.t_shape)
        _grad[self.index] = grad
        return _grad


class _Pow(Function):
    """pow 
    x^n -> n * (x ^ (n-1))
    n^x -> ln(y) = x*len(n) -> y' = y * ln(n)
    """
    def __init__(self, n):
        self.n = n 

    def forward(self, t):
        self.t = t
        return t ** self.n

    def backward(self, grad: Union[executor.support_types]):
        # grad * self.o * executor.np.log(self.t + 1e-15)
        return grad * self.n * (self.t ** (self.n-1))


class _RPow(Function):
    '''a ** x
    (a ** x) * log(a)
    '''
    def __init__(self, a):
        self.a = a

    def forward(self, t: Union[executor.support_types]):
        self.t = t
        self.out = self.a ** t 
        return self.out
    
    def backward(self, grad: Union[executor.support_types]):
        return grad * self.out * executor.np.log(self.a + 1e-10)


class _Exp(Function):
    """exp 
    """
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        self.out = executor.np.exp(t)
        return self.out
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        return grad * self.out


class _Sum(Function):
    ''' sum 
    '''
    def __init__(self, axis, keepdims):
        if isinstance(axis, int):
            axis = (axis, )
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, t: Union[executor.support_types]):
        self.t_shape = t.shape
        return t.sum(self.axis, keepdims=self.keepdims)

    def backward(self, grad: Union[executor.support_types]):
        if self.axis is None:
            self.axis = tuple(range(len(self.t_shape)))

        if self.keepdims:
            shape = grad.shape
        else:
            shape = list(self.t_shape)
            for ax in self.axis:
                shape[ax] = 1
        
        return grad.reshape(shape) * executor.np.ones(self.t_shape)


class _Mean(Function):
    ''' mean
    '''
    def __init__(self, axis, keepdims):
        if isinstance(axis, int):
            axis = (axis, )
        self.axis = axis 
        self.keepdims = keepdims

    def forward(self, t: Union[executor.support_types]):
        self.t_shape = t.shape
        return t.mean(self.axis, keepdims=self.keepdims)
    
    def backward(self, grad: Union[executor.support_types]):
        
        axis = self.axis
        if axis is None:
            axis = tuple(range(len(self.t_shape)))

        _shape = [(1 if i in axis else d) for i, d in enumerate(self.t_shape)]

        ks = [self.t_shape[i] for i in axis] if axis else [1]

        return grad.reshape(_shape) * executor.np.ones(self.t_shape) / REDUCE(MUL, ks)


class _Var(Function):
    '''sample variance
    '''
    def __init__(self, axis, keepdims, unbiased=True):
        if isinstance(axis, int):
            axis = (axis, )
        self.axis = axis 
        self.keepdims = keepdims
        self.unbiased = unbiased

    def forward(self, t):
        '''
        '''
        self.t_shape = t.shape

        if self.axis == None:
            self.n = REDUCE(MUL, self.t_shape)
        else:
            _shape = [d for i, d in enumerate(self.t_shape) if i in self.axis]
            self.n = REDUCE(MUL, _shape)

        t_minus_mean = t - t.mean(self.axis, keepdims=True)
        self.t_minus_mean = t_minus_mean

        if self.unbiased:
            return (t_minus_mean ** 2).sum(self.axis, keepdims=self.keepdims) / (self.n - 1)
        else:
            return (t_minus_mean ** 2).sum(self.axis, keepdims=self.keepdims) / self.n


    def backward(self, grad):
        '''
        '''
        _axis = self.axis
        if self.axis is None:
            _axis = tuple(range(len(self.t_shape)))
        
        _shape = [(1 if i in _axis else d) for i, d in enumerate(self.t_shape)]
        
        grad = grad.reshape(*_shape)

        if self.unbiased:
            return grad * 2 / (self.n - 1) * self.t_minus_mean
        else:
            return grad * 2 / self.n * self.t_minus_mean


class _Reshape(Function):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()
    
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        self.t_shape = t.shape
        return t.reshape(*self.shape)
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        grad = grad[...]
        return grad.reshape(*self.t_shape)


class _Transpose(Function):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, t: Union[executor.support_types]):
        assert len(self.dims) == len(t.shape)
        return t.transpose(*self.dims)
    
    def backward(self, grad: Union[executor.support_types]):
        idx_reverse = executor.np.argsort(self.dims)
        return grad.transpose(*idx_reverse)


class _Flip(Function):
    def __init__(self, *dims):
        self.dims = dims
    
    def forward(self, t):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError
    


# ------ register method

@register(Tensor)
def add(t, other):
    if isinstance(other, executor.support_types + (Tensor,)):
        other = to_tensor(other)
    else:
        return NotImplemented
        
    return _Add()(t, other)[0]


@register(Tensor)
def sub(t, other):
    if isinstance(other, executor.support_types + (Tensor,)):
        other = to_tensor(other)
    else:
        return NotImplemented

    return _Sub()(t, other)[0]

@register(Tensor)
def neg(t, ):
    return _Neg()(t)[0]

@register(Tensor)
def mul(t, other):
    other = to_tensor(other)
    return _Mul()(t, other)[0]

@register(Tensor)
def div(t, other):
    other = to_tensor(other)
    return _Div()(t, other)[0]

@register(Tensor)
def matmul(t, other):
    # other = to_tensor(other)
    return _Matmul()(t, other)[0]

@register(Tensor)
def pow(t, n):
    return _Pow(n)(t)[0]

@register(Tensor)
def sqrt(t, ):
    return _Pow(1/2.)(t)[0]

@register(Tensor)    
def exp(t, ):
    return _Exp()(t)[0]

@register(Tensor)
def rpow(t, a):
    return _RPow(a)(t)[0]

# statistas
@register(Tensor)
def sum(t, axis=None, keepdims=False):
    return _Sum(axis, keepdims)(t)[0]

@register(Tensor)
def mean(t, axis=None, keepdims=False):
    return _Mean(axis, keepdims)(t)[0]

# TODO
# @register(Tensor)
# def var(t, axis=None, keepdims=False):
#     return ((t - t.mean(axis, True)) ** 2).mean(axis, keepdims)

@register(Tensor)
def var(t, axis=None, keepdims=False, unbiased=True):
    return _Var(axis=axis, keepdims=keepdims, unbiased=unbiased)(t)[0]


# shape
@register(Tensor)
def reshape(t, *shape):
    return _Reshape(*shape)(t)[0]

@register(Tensor)
def transpose(t, *dims):
    return _Transpose(*dims)(t)[0]

@register(Tensor)
def t(t, ):
    dims = t.shape[::-1]
    return _Transpose(*dims)(t)[0]


@register(Tensor)
def flip(t, *dims):
    return _Flip(*dims)(t)[0]


# magic-method
@register(Tensor)
def __add__(self, other):
    '''self + other
    '''
    return self.add(other)

# @register(Tensor)
# def __radd__(self, other):
#     '''other + self
#     '''
#     # other = to_tensor(other)
#     # return other.add(self)
#     return self + other

@register(Tensor)
def __sub__(self, other):
    return self.sub(other)

@register(Tensor)
def __rsub__(self, other):
    # other = to_tensor(other)
    # return other.sub(self)
    return -self + other 

@register(Tensor)
def __neg__(self, ):
    return self.neg()

@register(Tensor)
def __mul__(self, other):
    return self.mul(other)

@register(Tensor)
def __rmul__(self, other):
    other = to_tensor(other)
    # return other.mul(self)
    return self * other

# @register(Tensor)
# def __div__(self, other):
#     return self.div(other)

# @register(Tensor)
# def __rdiv__(self, other):
#     other = to_tensor(other)
#     return other.div(self)

@register(Tensor)
def __truediv__(self, other):
    return self.div(other)

@register(Tensor)
def __rtruediv__(self, other):
    other = to_tensor(other)
    return other.div(self)

@register(Tensor)
def __matmul__(self, other):
    return self.matmul(other)

@register(Tensor)
def __pow__(self, n):
    return self.pow(n)

@register(Tensor)
def __rpow__(self, a):
    return self.rpow(a)

@register(Tensor)
def __getitem__(self, idx):
    return _GetItem(idx)(self)[0]



# ---- inspace-op

def gaurantee_inspace(var: Tensor):
    '''
    '''
    if isinstance(var.creator, Leaf) and var.requires_grad is True:
        raise RuntimeError('')

@register(Tensor)
def zeros_(self, ):
    gaurantee_inspace(self)
    self.storage[...] = 0

@register(Tensor)
def add_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] += other.storage
    elif isinstance(other, Union[executor.support_types]):
        self.storage[...] += other

@register(Tensor)
def sub_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] -= other.storage
    elif isinstance(other, Union[executor.support_types]):
        self.storage[...] -= other

@register(Tensor)
def mul_(self, other) -> None:
    gaurantee_inspace(self)
    if isinstance(other, ):
        self.storage[...] *= other.storage
    elif isinstance(other, Union[executor.support_types]):
        self.storage[...] *= other