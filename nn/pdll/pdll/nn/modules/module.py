import inspect
from collections import OrderedDict

from ..parameter import Parameter

class Module(object):
    '''Module
    '''
    def __init__(self, ):
        self._modules = OrderedDict()
        self._params = OrderedDict()
        self._buffers = OrderedDict()
    
    def __setattr__(self, key, value):
        '''
        '''
        if isinstance(value, Parameter):
            self._params[key] = value
        elif isinstance(value, Module):
            self._modules[key] = value
        else:
            pass
        object.__setattr__(self, key, value)

    def register_buffer(self, key, value):
        '''
        '''
        if key in self._buffers:
            raise RuntimeError
        self._buffers[key] = value

    def named_modules(self, modules=None, prefix=''):
        '''
        '''
        if modules is None:
            modules = set()
        if self not in modules:
            modules.add(self)
            yield prefix, self
        for name, module in self._modules.items():
            _prefix = prefix + ('.' if prefix else '') + name
            yield from module.named_modules(modules, _prefix)

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
        s = ''
        for n, m in self.named_modules():
            if isinstance(m, Module):
                s += f'{n} {m.__class__.__name__}{m.ext_repr()} \n'
        return s
    
    def ext_repr(self, ):
        return ''

    def train(self, ):
        '''train mode
        '''
        for name, value in inspect.getmembers(self):
            if name == 'training':
                self.__dict__['training'] = True
            elif isinstance(value, Module):
                value.train()

    def eval(self, ):
        '''eval mode
        '''
        for name, value in inspect.getmembers(self):
            if name == 'training':
                self.__dict__['training'] = False
            elif isinstance(value, Module):
                value.eval()


    def state_dict(self, ):
        '''
        '''
        raise NotImplementedError

    def load_state_dict(self, ):
        '''
        '''
        raise NotImplementedError