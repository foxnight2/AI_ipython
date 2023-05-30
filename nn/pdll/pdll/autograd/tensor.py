from typing import Union

from pdll.backend import executor 

from .backpropag import ExecuteEngine, Leaf


class Tensor(object):
    '''
    '''
    _engine = ExecuteEngine()
    
    def __init__(self, data, creator=None, requires_grad=False):
        assert isinstance(data, executor.support_types), f'{type(data)} not in {executor.support_types}'
        assert isinstance(requires_grad, bool), ''

        if creator is None:
            creator = Leaf(self, requires_grad)
        self._creator = creator
        self._storage = data
        self._grad = None

    @property
    def creator(self, ):
        return self._creator

    @property
    def data(self, ):
        return Tensor(self.storage)

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor), ''
        self = value

    @property
    def grad(self, ):
        return self._grad
    
    @grad.setter
    def grad(self, value):
        '''set grad
        '''
        assert isinstance(value, Tensor), ''
        self._grad = value

    @property
    def storage(self, ):
        return self._storage
    
    @storage.setter
    def storage(self, value):
        assert isinstance(value, executor.support_types), ''
        self._storage = value
    
    @property
    def shape(self, ):
        return self._storage.shape

    @property
    def dtype(self, ):
        return self._storage.dtype
        
    @property
    def requires_grad(self, ):
        return self.creator.requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        assert isinstance(value, bool), ''
        self.creator.requires_grad = value
    
    def numpy(self, ):
        '''numpy
        '''
        if executor.engine.engine_name == 'numpy':
            return self._storage[...]
        elif executor.engine.engine_name == 'cupy':
            return self._storage.get()
        else:
            raise RuntimeError

    def backward(self, grad=None):
        '''backward
        '''
        if grad is None:
            grad = self.__class__(executor.np.ones_like(self.storage))
        elif isinstance(grad, (int, float)):
            grad = self.__class__(executor.np.array([grad]))
        elif isinstance(grad, executor.support_types):
            grad = self.__class__(grad)
        elif isinstance(grad, self.__class__):
            assert grad.shape == self.shape, ''
        else:
            raise RuntimeError('type(grad) dont support.')
        
        self._engine.backward_fn(self.creator, grad)

    def zero_grad(self, ):
        if self.grad is not None:
            self.grad.storage[...] = 0

    def register_hook(self, name, hook):
        raise NotImplementedError

    def remove_hook(self, name):
        raise NotImplementedError 

    def __repr__(self, ):
        s = f'Tensor({self.storage}'
        if self.requires_grad:
            s += f', requires_grad={self.requires_grad})'
        else:
            s += ')'
        return s

    def __array__(self):
        return self.numpy()
        