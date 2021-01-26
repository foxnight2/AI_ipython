import numpy as np
from numpy import ndarray as Tensor 

from typing import Any, Tuple, Optional
from collections import OrderedDict, Counter

from functools import reduce
import operator
import inspect
import math


class Function(object):

    def __init__(self, ):
        self.previous_functions = None
        self.output_ids = None
        self.needs_input_grad = None
        self.backward_hooks = OrderedDict()

    def _do_forward(self, *inputs):
        '''
        '''
        self.inputs = inputs # for backword

        unpacked_input = tuple(arg.data for arg in inputs)
        # unpacked_input = []
        # for var in inputs:
        #     if isinstance(var, Variable):
        #         unpacked_input.append(var.data)
        #     else:
        #         unpacked_input.append(var)

        raw_output = self.forward(*unpacked_input)

        if not isinstance(raw_output, tuple):
            raw_output = (raw_output, )
        
        self.needs_input_grad = tuple(arg.creator.requires_grad for arg in inputs)
        self.requires_grad = any(self.needs_input_grad)

        output = tuple(Variable(data, self) for data in raw_output)

        self.previous_functions = [(arg.creator, id(arg)) for arg in inputs]
        self.output_ids = {id(var): i for i, var in enumerate(output)}

        return output

    __call__ = _do_forward

    def _do_backward(self, grad_output):
        '''
        '''
        grad_input = self.backward(grad_output) 
        if not isinstance(grad_input, tuple):
            grad_input = (grad_input, )
        
        assert len(grad_input) == len(self.previous_functions), f'{self.__class__.__name__}'

        return grad_input


    def register_hook(self, name, hook, tensor):
        assert name not in self.backward_hooks, ''
        ids = self.output_ids[id(tensor)] if tensor else None
        self.backward_hooks[name] = (hook, ids)

    def remove_hook(self, name):
        assert name in self.backward_hooks, ''
        del self.backward_hooks[name]
    
    def forward(self, *inputs):
        '''tensor -> tensor
        '''
        raise NotImplementedError

    def backward(self, *grad_output):
        '''tensor -> tensor
        '''
        raise NotImplementedError



class ExecuteEngine(object):
    
    def __init__(self, ):
        pass

    def _compute_depends(self, function):
        depends = {}
        seen = {function}
        queue = [function]

        while len(queue) > 0:

            fn = queue.pop()
            for prev_fn, arg_id in fn.previous_functions:
                if prev_fn not in depends:
                    depends[prev_fn] = [Counter() for _ in prev_fn.output_ids]
                output_idx = prev_fn.output_ids[arg_id]
                depends[prev_fn][output_idx][fn] += 1
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)

        return depends


    def _free_backward_depend(self, depends, prev_fn, fn, arg_id):
        deps = depends[prev_fn]
        output_idx = prev_fn.output_ids[arg_id]
        output_deps = deps[output_idx]
        output_deps[fn] -= 1
        if output_deps[fn] == 0:
            del output_deps[fn]
        return output_idx


    def _is_ready_for_backward(self, depends, functiton):
        for deps in depends[functiton]:
            if len(deps) > 0:
                return False
        return True


    def run_backward(self, variable, grad):
        
        ready = [(variable.creator, (grad, ))]
        not_ready = []

        depends = self._compute_depends(variable.creator)
        
        while len(ready) > 0:
            
            fn, grad = ready.pop()
            grad_input = fn._do_backward(*grad)

            for (prev_fn, arg_id), d_prev_fn in zip(fn.previous_functions, grad_input):
                if not prev_fn.requires_grad:
                    assert d_prev_fn is None
                    continue

            output_nr = self._free_backward_depend(depends, prev_fn, fn, arg_id)
            is_ready = self._is_ready_for_backward(depends, prev_fn)
                
        raise NotImplementedError


    def _backward_var(self, var, grad):
        ''' '''
        var.grad += grad
        grads_input = var.creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            # var.creator.previous_functions[_i][0]._do_backward(_grad)
            self._backward_var(var.creator.inputs[_i], _grad)

    def _backward_fn(self, creator, grad):
        grads_input = creator._do_backward(grad)
        for _i, _grad in enumerate(grads_input):
            if _grad is not None:
                self._backward_fn(creator.previous_functions[_i][0], _grad)


# --- utils
def to_tensor(data):
    return np.array(data).astype(np.float64)

def to_variable(data):
    '''make sure data is variable'''
    if isinstance(data, (int, float)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, (list, tuple)):
        data = to_tensor(data)
        return Variable(data)

    elif isinstance(data, Tensor):
        return Variable(data)
        
    elif isinstance(data, Variable):
        return data
# ---


class Variable(object):

    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        if creator is None:
            creator = Leaf(self, requires_grad)
        self.data = data
        self.creator = creator
        self.shape = self.data.shape
        self.requires_grad = self.creator.requires_grad
        
        self.grad = None
        if isinstance(creator, Leaf) and requires_grad:
            self.grad = np.zeros_like(data)

    def backward(self, grad=1.):
        self._engine._backward_fn(self.creator, grad)

    def zero_grad(self, ):
        self.grad[...] = 0

    def register_hook(self, name, hook):
        self.creator.register_hook(name, hook)
    
    def remove_hook(self, name):
        self.creator.remove_hook(name)

    def reshape(self, *shape):
        return Reshape(*shape)(self)[0]


    def add(self, other):
        other = to_variable(other)
        return Add()(self, other)[0]
    
    def neg(self, ):
        return Neg()(self)[0]

    def sub(self, other):
        other = to_variable(other)
        # return self.add(other.neg())
        return Sub()(self, other)[0]

    def mul(self, other):
        other = to_variable(other)
        return Mul()(self, other)[0]

    def div(self, other):
        other = to_variable(other)
        return Div()(self, other)[0]

    def sum(self, ):
        return Sum()(self)[0]
    
    def mean(self, ):
        return Mean()(self)[0]

    def pow(self, n):
        n = to_variable(n)
        return Pow()(self, n)[0]

    def exp(self, ):
        return Exp()(self)[0]

    def sigmoid(self, ):
        return op_sigmoid()(self)[0]

    def tanh(self, ):
        return op_tanh()(self)[0]

    def matmul(self, other: 'Variable') -> 'Variable':
        return Matmul()(self, other)[0]
    
    def t(self, ):
        raise NotImplementedError

    @staticmethod
    def T(self, ):
        return self.t()

    def transpose(self, *dims):
        return Transpose(*dims)(self)[0]

    # magic method
    def __add__(self, other):
        return self.add(other)

    def __neg__(self, ):
        return self.neg()

    def __sub__(self, other):
        return self.sub(other)

    def __pow__(self, n):
        return self.pow(n)
    
    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)
    
    # __radd__ = __add__
    def __radd__(self, other):
        other = to_variable(other)
        return other.add(self)
        
    def __rsub__(self, other):
        other = to_variable(other)
        return other.sub(self)
    
    def __rmul__(self, other):
        other = to_variable(other)
        return other.mul(self)

    def __rdiv__(self, other):
        other = to_variable(other)
        return other.div(self)
    __rtruediv__ = __rdiv__


    def __iadd__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __getitem__(self, idx):
        return Getitem(idx)(self)[0]


# ====
class Parameter(Variable):

    def __init__(self, *shape, data=None):
        if shape:
            data = np.random.rand(shape) * 2 - 1
        super().__init__(data, requires_grad=True)


# ====
class Leaf(Function):

    def __init__(self, variable, requires_grad):
        self.variable = variable
        self.output_ids = {id(variable): 0}
        self.previous_functions = []
        self.requires_grad = requires_grad
        # self.backward_hooks = OrderedDict()

    def _do_forward(self, *input):
        raise NotImplementedError

    def _do_backward(self, *grad_output):
        assert len(grad_output) == 1
        if self.requires_grad:
            self.variable.grad += grad_output[0]
        return tuple()


def broadcast_reverse(grad: Tensor, shape: Tuple[int]) -> Tensor: 
    '''reverse grad to shape
    '''
    _extdims = grad.ndim - len(shape)
    for _ in range(_extdims):
        grad = grad.sum(axis=0)
    assert len(grad.shape) == len(shape), ''

    for i, d in enumerate(shape):
        if d == 1:
            grad = grad.sum(axis=i, keepdims=True)
    assert grad.shape == shape, ''
    
    return grad


# operations

class Add(Function):
    """add
    broadcast
    [1, 3] + [2, 4, 3] -> [2, 4, 3]
    """
    def forward(self, a, b):
        c = a + b
        self.a_shape = a.shape
        self.b_shape = b.shape
        self.c_shape = c.shape
        return c

    def backward(self, grad):
        assert self.c_shape == grad.shape, ''
        a_grad = broadcast_reverse(grad, self.a_shape)
        b_grad = broadcast_reverse(grad, self.b_shape)

        return a_grad, b_grad


class Neg(Function):
    """
    -t 
    """
    def forward(self, t: Tensor) -> Tensor:
        return -t 
    
    def backward(self, grad: Tensor) -> Tensor:
        return -grad


class Sub(Function):
    """a-b
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.a_shape = a.shape
        self.b_shape = b.shape
        return a - b

    def backward(self, grad):
        a_grad = broadcast_reverse( grad, self.a_shape)
        b_grad = broadcast_reverse(-grad, self.b_shape)
        return a_grad, b_grad 


class Mul(Function):
    """MUL"""
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        c = a * b
        self.a = a
        self.b = b
        self.c_shape = c.shape
        return c
    
    def backward(self, grad):
        assert self.c_shape == grad.shape
        a_grad = broadcast_reverse(grad * self.b, self.a.shape)
        b_grad = broadcast_reverse(grad * self.a, self.b.shape)

        return a_grad, b_grad


class Div(Function):
    """div
    """
    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # np.testing.assert_almost_equal(b, 0)
        self.a = a
        self.b = b
        return a / (b + 1e-20)
    
    def backward(self, grad: Tensor):
        a_grad = grad / self.b
        b_grad = -grad * self.a / (self.b ** 2)
        return a_grad, b_grad


class Sum(Function):
    """ sum """
    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t.sum()

    def backward(self, grad: Tensor):
        print('mean')
        return grad * np.ones(self.t_shape)


class Mean(Function):
    """ mean """
    def forward(self, t: Tensor):
        self.t = t
        return t.mean()
    
    def backward(self, grad: Tensor):
        return grad * np.ones_like(self.t) / reduce(operator.mul, self.t.shape)


class Pow(Function):
    """pow 
    x^n -> n * (x ^ (n-1))
    n^x -> ln(y) = x*len(n) -> y' = y * ln(n)
    """
    def forward(self, t, n):
        self.t = t
        self.n = n
        self.o = t ** n
        return self.o

    def backward(self, grad: Tensor):
        return grad * self.n * (self.t ** (self.n-1)), None # grad * self.o * np.log(self.t + 1e-15)


class Exp(Function):
    """exp """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.exp(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.out


class Matmul(Function):
    """
    t1 @ t2 [2, 3] [3, 5] -> [2, 5]
    grad @ t2.T [2, 5] [5, 3] -> [2, 3]
    t1.T @ grad [3, 2] [2, 5] -> [3, 5]
    """
    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        self.t1 = t1
        self.t2 = t2
        return t1 @ t2
    
    def backward(self, grad: Tensor) -> Tuple[Tensor]:
        return grad @ self.t2.T, self.t1.T @ grad
    

class Getitem(Function):
    """getitem"""
    def __init__(self, index):
        self.index = index
        super().__init__()
    
    def forward(self, t: Tensor):
        self.t_shape = t.shape
        return t[self.index]
    
    def backward(self, grad):
        _grad = np.zeros(shape=self.t_shape)
        _grad[self.index] = grad
        return _grad


class Reshape(Function):
    def __init__(self, *shape):
        self.shape = shape
        super().__init__()
    
    def forward(self, t: Tensor) -> Tensor:
        self.t_shape = t.shape
        return t.reshape(*self.shape)
    
    def backward(self, grad: Tensor) -> Tensor:
        # grad = grad[...]
        return grad.reshape(*self.t_shape)


class Transpose(Function):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, t: Tensor):
        assert len(self.dims) == len(t.shape)
        return t.transpose(*self.dims)
    
    def backward(self, grad: Tensor):
        idx_reverse = np.argsort(self.dims)
        return grad.reshape(*idx_reverse)


## ----Activation

class op_sigmoid(Function):
    """sigmoid """
    def forward(self, t: Tensor):
        self.out = 1. / (1. + np.exp(-t)) 
        return self.out
    
    def backward(self, grad):
        return grad * self.out / (1. - self.out + 1e-10)


class op_relu(Function):
    """relu """ 
    def forward(self, t: Tensor) -> Tensor:
        self.mask = t > 0
        return t * self.mask
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.mask


class op_tanh(Function):
    """
    formul: (exp(x) + exp(-x)) / (exp(x) - exp(-x))
    derive : 1 - tanh(x) ** 2
    """
    def forward(self, t: Tensor) -> Tensor:
        self.out = np.tanh(t)
        return self.out
    
    def backward(self, grad: Tensor) -> Tensor:
        return grad  * (1 - self.out ** 2)


# --- Conv

def im2col(data, kernel, stride, padding):
    '''im2col
    N C H W -> n h w c k k
    '''
    n, c, h, w = data.shape
    out_h = math.floor((h + 2 * padding[0] - kernel[0]) / stride[0] + 1)
    out_w = math.floor((w + 2 * padding[1] - kernel[1]) / stride[1] + 1)

    # hpad = (padding[0]//2, padding[0] - padding[0]//2)
    # wpad = (padding[1]//2, padding[1] - padding[1]//2)
    hpad = (padding[0], padding[0])
    wpad = (padding[1], padding[1])
    data = np.pad(data, pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant')
    
    matrix = np.zeros((n, c, kernel[0], kernel[1], out_h, out_w))

    for i in range(kernel[0]):
        iend = i + stride[0] * out_h
        for j in range(kernel[1]):
            jend = j + stride[1] * out_w
            matrix[:, :, i, j, :, :] = data[:, :, i:iend:stride[0], j:jend:stride[1]]
            # matrix[:, :, i, j, :, :] = data[:, :, i::stride[0], j::stride[1]]
        
    return matrix, out_h, out_w
    

def col2im(matrix, shape, kernel, stride, padding):
    '''
    matrix  n, ho, wo, cin, hk, wk
    '''
    _, _, _, _, ho, wo = matrix.shape
    # matrix = matrix.transpose(0, 3, 4, 5, 1, 2) # (n, c, hk, wk, ho, wo)

    hpad = (padding[0], padding[0])
    wpad = (padding[1], padding[1])
    data = np.pad(np.zeros(shape), pad_width=((0, 0), (0, 0), hpad, wpad), mode='constant',)
    _, _, H, W = data.shape
    
    for i in range(kernel[0]):
        iend = i + stride[0] * ho
        for j in range(kernel[1]):
            jend = j + stride[1] * wo
            data[:, :, i:iend:stride[0], j:jend:stride[1]] += matrix[:, :, i, j, :, :]

    return data[:, :, padding[0]:H-padding[0], padding[1]:W-padding[1]]


class op_conv2d(Function):
    '''conv
    '''
    def __init__(self, kernel, stride, padding):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, data: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        '''
        n c h w
        co ci kh kw
        '''
        # self.data = data
        self.weight = weight
        self.data_shape = data.shape

        n, c, _, _ = data.shape
        c_out, _, _, _ = weight.shape
        
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding) # -> n*hout*wout cin*hk*wk
        matrix = matrix.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, c * self.kernel[0] * self.kernel[1])

        weight = weight.transpose(1, 2, 3, 0).reshape(-1, c_out) # -> cin*hk*wk cout
        
        self.matrix = matrix

        output = (matrix @ weight).reshape(n, out_h, out_w, c_out).transpose(0, 3, 1, 2)
        
        if bias is not None:
            return output + bias.reshape(1, -1, 1, 1)
        else:
            return output


    def backward(self, grad: Tensor):
        '''grad n cout hout wout
        '''
        n, cout, hout, wout = grad.shape
        _, cin, hk, wk = self.weight.shape

        bias_grad = grad.sum(axis=(0, 2, 3))

        # indx_reverse = np.argsort([0, 3, 1, 2])
        grad_reverse = grad.transpose(0, 2, 3, 1)
        grad_reverse = grad_reverse.reshape(n * hout * wout, cout)
        
        weight_grad = self.matrix.T @ grad_reverse # cin hk wk cout
        weight_grad = weight_grad.reshape(cin, hk, wk, cout)
        weight_grad = weight_grad.transpose(3, 0, 1, 2)

        weight = self.weight.transpose(1, 2, 3, 0).reshape(-1, cout)  # -> cin*hk*wk cout
        data_grad = grad_reverse @ weight.T # n*hout*wout cin*hk*wk
        data_grad = data_grad.reshape(n, hout, wout, cin, hk, wk)
        data_grad = data_grad.transpose(0, 3, 4, 5, 1, 2) # (n, cin, hk, wk, hout, wout)
        data_grad = col2im(data_grad, self.data_shape, self.kernel, self.stride, self.padding)

        return data_grad, weight_grad, bias_grad


class op_pool2d(Function):
    def __init__(self, kernel, stride, padding, mode='max'):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.mode = mode

    def forward(self, data: Tensor):
        ''''''
        self.shape = data.shape
        n, c, _, _ = data.shape
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding)
        matrix = matrix.reshape(n, c, self.kernel[0] * self.kernel[1], out_h, out_w)
        self.matrix = matrix

        if self.mode.lower() == 'max':
            out = np.max(matrix, axis=2)
        elif self.mode.lower() == 'avg':
            out = np.average(matrix, axis=2)
        else:
            raise RuntimeError
    
        return out


    def backward(self, grad: Tensor):
        n, c, oh, ow = grad.shape
        grad = grad[:, :, np.newaxis, :, :]
        if self.mode.lower() == 'max':
            mask = self.matrix == np.max(self.matrix, axis=2, keepdims=True)
            grad = grad * mask
        elif self.mode.lower() == 'avg':
            grad = grad * np.ones_like(self.matrix) / (self.kernel[0] * self.kernel[1])
        else:
            raise RuntimeError

        grad = grad.reshape(n, c, self.kernel[0], self.kernel[1], oh, ow)

        return col2im(grad, self.shape, self.kernel, self.stride, self.padding)


# --- Module

class Module(object):

    def named_parameters(self, ):
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield name, value
            elif isinstance(value, Module):
                yield from value.named_parameters()
            else:
                pass
        
    def parameters(self, ):
        for _, value in self.named_parameters():
            yield value

    def zero_grad(self, ):
        for p in self.parameters():
            p.zero_grad()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __repr__(self, ):
        '''str
        '''
        s = self.__class__.__name__ + self.ext_repr()

        for n, m in inspect.getmembers(self):
            if isinstance(m, Module):
                _s = f'\n  {n} {str(m)}'
                s += _s

        return s
    
    def ext_repr(self, ):
        return ''


class Tanh(Module):
    '''tanh
    '''
    def __init__(self, ):
        pass

    def forward(self, data):
        return op_tanh()(data)[0]

    def ext_repr(self, ):
        return ''


class Simoid(Module):
    '''sigmoid
    ''' 
    def forward(self, data):
        return op_sigmoid()(data)[0]
    
    def ext_repr(self, ):
        return ''


class ReLU(Module):
    '''relu
    '''
    def forward(self, data):
        return op_relu()(data)[0]

    def ext_repr(self, ):
        return ''


class Linear(Module):
    """Linear 
    """
    def __init__(self, input_dim, output_dim, bias=True):
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        k = math.sqrt(1. / input_dim)
        init_weight = np.random.uniform(low=-k, high=k, size=(input_dim, output_dim))
        self.weight = Parameter(data=init_weight)
        if self.bias:
            init_bias = np.random.uniform(low=-k, high=k, size=(output_dim, ))
            self.bias = Parameter(data=init_bias)
        
    def forward(self, data):
        if self.bias:
            return data @ self.weight + self.bias
        else:
            return data @ self.weight

    def ext_repr(self, ):
        return f'({self.input_dim}, {self.output_dim})'    


class Conv2d(Module):
    '''
    image: C_in H_in W_in
    kernel: C_out C_in H_kernel W_kernel
    output: C_out H_out W_out
    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1)
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.groups = groups
        
        assert in_channels % groups == 0, ''

        k = math.sqrt(1. / (groups * in_channels * kernel_size[0] * kernel_size[1]))
        weight_init = np.random.uniform(low=-k, high=k, size=(out_channels, int(in_channels/groups), kernel_size[0], kernel_size[1]))
        self.weight = Parameter(data=weight_init)
        
        if bias:
            bias_init = np.random.uniform(low=-k, high=k, size=(self.out_channels, ))
            self.bias = Parameter(data=bias_init)
        else:
            self.bias = None

    def forward(self, data):
        return op_conv2d(self.kernel_size, self.stride, self.padding)(data, self.weight, self.bias)[0]

    def ext_repr(self, ):
        return f'({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})'



class Pool2d(Module):
    def __init__(self, kernel_size, stride, padding, mode='max'):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.mode = mode

    def forward(self, data: Tensor):
        return op_pool2d(self.kernel_size, self.stride, self.padding, self.mode)(data)[0]

    def ext_repr(self, ):
        return f'(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, mode={self.mode})'


