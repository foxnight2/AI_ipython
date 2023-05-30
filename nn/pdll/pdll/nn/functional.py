import math
from typing import Tuple, Union
import operator
import functools

from pdll.backend import executor

from pdll.autograd import Function, Tensor, register

from .parameter import Parameter
from .utils import im2col, col2im


class _Sigmoid(Function):
    """sigmoid
    """
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        self.out = 1. / (1. + executor.np.exp(-t)) 
        return self.out
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        return grad * self.out * (1. - self.out)


class _ReLU(Function):
    """relu 
    """ 
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        self.mask = t > 0
        return t * self.mask
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        return grad * self.mask


class _Tanh(Function):
    """
    formul: (exp(x) + exp(-x)) / (exp(x) - exp(-x))
    derive : 1 - tanh(x) ** 2
    """
    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        self.out = executor.np.tanh(t)
        return self.out
    
    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        return grad  * (1 - self.out ** 2)


class _Mish(Function):
    '''Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681v1
    x * tanh( softplus(x) )
    '''
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = 1
        self.threshold = 20

    def forward(self, x):
        self.mask = self.beta * x > self.threshold
        self.x = x 

        dummy = x * (1 - self.mask)
        xexp = 1 + executor.np.exp(self.beta * dummy)
        dummy = 1 / self.beta * executor.np.log(xexp)
        dummy[self.mask] = x[self.mask]

        self.xexp = xexp
        # self.softplus = dummy
        self.tanhsp = executor.np.tanh(dummy)

        return x * self.tanhsp

    def backward(self, grad):
        dx = self.tanhsp + self.x * (1 - self.tanhsp ** 2) * (1 - 1 / self.xexp)
        dx[self.mask] = (self.tanhsp + self.x * (1 - self.x ** 2))[self.mask]

        return grad * dx 


class _Softplus(Function):
    '''
    forward: 1 / beta * log(1 + exp(beta * x))
    backward: grad * (1 - 1. / (1 + exp(beta * x)))
    '''
    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.beta = 1
        self.threshold = 20

    def forward(self, x):
        self.mask = self.beta * x > self.threshold

        dummy = x * (1 - self.mask)
        xexp = 1 + executor.np.exp(self.beta * dummy)
        dummy = 1 / self.beta * executor.np.log(xexp)
        dummy[self.mask] = x[self.mask]

        self.xexp = xexp

        return dummy

    def backward(self, grad):
        dx = 1 - 1 / self.xexp
        dx[self.mask] = 1. 
        return grad * dx


@register(Tensor)
def relu(self, ):
    return _ReLU()(self)[0]

@register(Tensor)
def tanh(self, ):
    return _Tanh()(self)[0]

@register(Tensor)
def sigmoid(self, ):
    return _Sigmoid()(self)[0]

@register(Tensor)
def softplus(self, ):
    return _Softplus(beta=1, threshold=20)(self)[0]

@register(Tensor)
def mish(self, ):
    return _Mish(beta=1, threshold=20)(self)[0]

class _Conv2d(Function):
    '''conv
    '''
    def __init__(self, kernel, stride, padding, dilation, groups):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, data: Union[executor.support_types], weight: Union[executor.support_types], bias: Union[executor.support_types]) -> Union[executor.support_types]:
        '''
        n c h w
        co ci kh kw
        '''
        # self.data = data
        self.weight = weight
        self.data_shape = data.shape
        
        n, cin, _, _ = data.shape
        cout, _, _, _ = weight.shape
        
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding, self.dilation) # -> n*hout*wout cin*hk*wk
        
        # matrix = matrix.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, cin * self.kernel[0] * self.kernel[1])
        # weight = weight.transpose(1, 2, 3, 0).reshape(-1, cout) # -> cin*hk*wk cout  [groups * cout/groups * cin
        # self.matrix = matrix
        # output = (matrix @ weight).reshape(n, out_h, out_w, cout).transpose(0, 3, 1, 2)

        matrix = matrix.reshape(n, self.groups, cin//self.groups, self.kernel[0], self.kernel[1], out_h, out_w)
        matrix = matrix.transpose(1, 0, 5, 6, 2, 3, 4).reshape(self.groups, n * out_h * out_w, cin//self.groups * self.kernel[0] * self.kernel[1])

        weight = weight.reshape(self.groups, cout//self.groups, cin//self.groups, self.kernel[0], self.kernel[1])
        weight = weight.transpose(0, 2, 3, 4, 1).reshape(self.groups, cin//self.groups * self.kernel[0] * self.kernel[1], cout//self.groups)
        # output = matrix.bmm(weight) # groups n*out_h*out_w cout/groups
        
        self.matrix = matrix # groups n*hout*wout cin//groups*hk*wk
        self.weight = weight # groups cin//groups*hk*wk cout//groups
        output = (matrix @ weight).transpose(1, 0, 2).reshape(n * out_h * out_w, self.groups * cout//self.groups)
        output = output.reshape(n, out_h, out_w, cout).transpose(0, 3, 1, 2)
        
        if bias is not None:
            return output + bias.reshape(1, -1, 1, 1)
        else:
            return output


    def backward(self, grad: Union[executor.support_types]):
        '''grad n cout hout wout
        '''
        n, cout, hout, wout = grad.shape
        # _, cin, hk, wk = self.weight.shape
        _, cin, _, _ = self.data_shape

        bias_grad = grad.sum(axis=(0, 2, 3))

        # indx_reverse = executor.np.argsort([0, 3, 1, 2])
        # grad_reverse = grad.transpose(0, 2, 3, 1)
        # grad_reverse = grad_reverse.reshape(n * hout * wout, cout)
        
        # weight_grad = self.matrix.T @ grad_reverse # cin hk wk cout
        # weight_grad = weight_grad.reshape(cin, hk, wk, cout)
        # weight_grad = weight_grad.transpose(3, 0, 1, 2)

        # weight = self.weight.transpose(1, 2, 3, 0).reshape(-1, cout)  # -> cin*hk*wk cout
        # data_grad = grad_reverse @ weight.T # n*hout*wout cin*hk*wk
        # data_grad = data_grad.reshape(n, hout, wout, cin, hk, wk)
        # data_grad = data_grad.transpose(0, 3, 4, 5, 1, 2) # (n, cin, hk, wk, hout, wout)
        # data_grad = col2im(data_grad, self.data_shape, self.kernel, self.stride, self.padding)

        grad_reverse = grad.transpose(0, 2, 3, 1) # n, hout, wout, cout
        grad_reverse = grad_reverse.reshape(n * hout * wout, self.groups, cout//self.groups)
        grad_reverse = grad_reverse.transpose(1, 0, 2) # groups, n*hout*wout, cout//groups
        
        weight_grad = self.matrix.transpose(0, 2, 1) @ grad_reverse # bmm
        weight_grad = weight_grad.reshape(self.groups, cin//self.groups*self.kernel[0]*self.kernel[1], cout//self.groups)
        weight_grad = weight_grad.transpose(0, 2, 1).reshape(cout, cin//self.groups, self.kernel[0], self.kernel[1])

        data_grad = grad_reverse @ self.weight.transpose(0, 2, 1) # groups, n*hout*wout, cin//groups*hk*wk
        data_grad = data_grad.transpose(1, 0, 2).reshape(n * hout * wout, cin * self.kernel[0] * self.kernel[1])
        data_grad = data_grad.reshape(n, hout, wout, cin, self.kernel[0], self.kernel[1])
        data_grad = data_grad.transpose(0, 3, 4, 5, 1, 2)
        data_grad = col2im(data_grad, self.data_shape, self.kernel, self.stride, self.padding)

        return data_grad, weight_grad, bias_grad


def conv2d(v: Tensor, w: Parameter, b: Parameter, kernel: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding: Union[int, Tuple[int, ...]], dilation: int, groups: int):
    '''conv2d
    '''
    return _Conv2d(kernel, stride, padding, dilation, groups)(v, w, b)[0]



class _Pool2d(Function):
    '''pooling
    '''
    def __init__(self, kernel, stride, padding, dilation=1, mode='max'):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mode = mode

    def forward(self, data: Union[executor.support_types]):
        ''''''
        self.shape = data.shape
        n, c, _, _ = data.shape
        matrix, out_h, out_w = im2col(data, self.kernel, self.stride, self.padding, dilation=self.dilation)
        matrix = matrix.reshape(n, c, self.kernel[0] * self.kernel[1], out_h, out_w)
        self.matrix = matrix

        if self.mode.lower() == 'max': # TODO
            out = executor.np.max(matrix, axis=2)
        elif self.mode.lower() == 'avg':
            out = executor.np.average(matrix, axis=2)
        else:
            raise RuntimeError
    
        return out


    def backward(self, grad: Union[executor.support_types]):
        n, c, oh, ow = grad.shape
        grad = grad[:, :, executor.np.newaxis, :, :]
        if self.mode.lower() == 'max':
            mask = self.matrix == executor.np.max(self.matrix, axis=2, keepdims=True)
            grad = grad * mask
        elif self.mode.lower() == 'avg':
            grad = grad * executor.np.ones_like(self.matrix) / (self.kernel[0] * self.kernel[1])
        else:
            raise RuntimeError

        grad = grad.reshape(n, c, self.kernel[0], self.kernel[1], oh, ow)

        return col2im(grad, self.shape, self.kernel, self.stride, self.padding, dilation=self.dilation)



def pool2d(v: Tensor, kernel: Union[int, Tuple[int, ...]], stride: Union[int, Tuple[int, ...]], padding: Union[int, Tuple[int, ...]]=0, dilation: int=1, mode: str='max') -> Tensor:
    '''pool2d
    '''
    if isinstance(kernel, int):
        kernel = (kernel, kernel)

    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    elif isinstance(padding, (tuple, list)) and len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    elif isinstance(padding, (tuple, list)) and len(padding) == 4:
        padding = tuple(padding)
    else:
        raise RuntimeError('not suppot padding format')

    return _Pool2d(kernel, stride, padding, dilation, mode)(v)[0]

max_pool2d = functools.partial(pool2d, mode='max')
avg_pool2d = functools.partial(pool2d, mode='avg')


class _Softmax(Function):
    
    def __init__(self, axis: int):
        self.axis = axis
        
    def forward(self, data):
        '''softmax(x-c)
        -c: deal with overflow inf problem
        '''
        t = executor.np.exp(data - data.max(axis=self.axis, keepdims=True))
        a = t / (t.sum(axis=self.axis, keepdims=True))
        self.a = a 
        return a

    def backward(self, grad):
        '''
        dl/da_i da_i/exp(x_i)
        dl/da_k da_k/sum_exp(x_k)
        dl/dx_i = dl/da_i * a_i - a_i * sum_j(dl/da_j * a_j)
        '''
        return self.a * (grad - (grad * self.a).sum(axis=self.axis, keepdims=True))
        

@register(Tensor)
def softmax(self: Tensor, axis: int) -> Tensor:
    '''softmax
    '''
    return _Softmax(axis)(self)[0]


class _CrossEntropy(Function):
    
    def __init__(self, axis=-1, reduction='mean'):
        super().__init__()
        self.axis = axis
        self.reduction = reduction.lower()

    def forward(self, logit, label):
        '''label one-hot
        '''
        t = executor.np.exp(logit - logit.max(axis=self.axis, keepdims=True))
        a = t / (t.sum(axis=self.axis, keepdims=True))

        self.a = a
        self.label = label

        if self.reduction == 'sum':
            return (-label * executor.np.log(a)).sum()

        elif self.reduction == 'mean':
            return (-label * executor.np.log(a)).sum() / functools.reduce(operator.mul, logit.shape[:-1])
        
    def backward(self, grad=1.):
        '''grad = 1.
        '''
        if self.reduction == 'sum':
            grad = grad * executor.np.ones_like(self.a)
        elif self.reduction == 'mean':
            grad = grad / functools.reduce(operator.mul, self.a.shape[:-1]) * executor.np.ones_like(self.a)
            
        grad_logit = grad * (self.a - self.label)

        return grad_logit, None


def cross_entropy(logit: Tensor, label: Tensor, axis: int=-1, reduction: str='mean') -> Tensor:
    '''
    '''
    return _CrossEntropy(axis, reduction)(logit, label)[0]


# 
class _Dropout(Function):
    '''
    '''
    def __init__(self, p: float, training: bool=True, inspace: bool=True):
        self.p = p 
        self.training = training
        self.inspace = inspace

    def forward(self, t: Union[executor.support_types]) -> Union[executor.support_types]:
        '''
        '''
        if not self.training:
            self.p = 0. 
        mask = executor.np.random.rand(*t.shape) > self.p
        self.mask = mask 

        return t * mask / (1 - self.p)

    def backward(self, grad):
        '''
        '''
        return grad * self.mask / (1 - self.p)


def dropout(v: Tensor, p: float, training: bool=True, inspace: bool=True) -> Tensor:
    '''dropout
    '''
    return _Dropout(p, training, inspace)(v)[0]



class _Padding(Function):
    '''
    '''
    def __init__(self, pad: Union[int, Tuple[int, ...]], mode: str='constant', value: float=0):
        super().__init__()
        if isinstance(pad, int):
            pad = (pad, )
        else:
            assert len(pad) % 2 == 0, ''
        self.pad = tuple(pad)
        self.mode = mode
        self.value = value
        assert self.mode in ('constant')

    def forward(self, data: Union[executor.support_types]) -> Union[executor.support_types]:
        '''
        '''
        pad = self.pad
        shape = data.shape

        assert len(shape) >= len(pad)//2, ''
        if len(pad) == 1:
            pad = tuple([pad[0], ] * (2 * len(shape)))
        else:
            pad = pad + (0, ) * (2 * len(shape) - len(pad))
        
        assert len(pad) == 2 * len(shape), ''
        
        padding = list(zip(pad[0::2][::-1], pad[1::2][::-1]))

        self.shape = shape
        self.padding = padding

        return executor.np.pad(data, pad_width=padding, mode=self.mode, constant_values=self.value)

    def backward(self, grad: Union[executor.support_types]) -> Union[executor.support_types]:
        '''
        '''
        slices = []
        for pad in self.padding:
            if pad[1] == 0:
                slices.append(slice(pad[0], None))
            else:
                slices.append(slice(pad[0], -pad[1]))

        return grad[tuple(slices)]


def zero_pad2d(data: Tensor, padding: Union[int, Tuple[int, int, int, int]]):
    '''zero pad2d
    '''
    assert len(data.shape) == 4, ''
    if isinstance(padding, int):
        padding = (padding, ) * 4

    return _Padding(padding, mode='constant', value=0)(data)[0]


def constant_pad2d(data: Tensor, padding: Union[int, Tuple[int, int, int, int]], value: float=0):
    '''constant pad2d
    '''
    assert len(data.shape) == 4, ''
    if isinstance(padding, int):
        padding = (padding, ) * 4
    return _Padding(padding, mode='constant', value=value)(data)[0]


def attention(query: Tensor, key: Tensor, value: Tensor, key_mask=None, att_mask=None, dropout=None):
    ''' [n * h, l, dim]
    '''
    dim = query.shape[-1]
    score = query @ key.transpose(0, 2, 1) / math.sqrt(dim) # [n * h, l_q, l_s]

    att_score = softmax(score, axis=-1)

    if dropout:
        att_score = dropout(att_score)

    return att_score @ value, att_score