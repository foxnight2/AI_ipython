import torch
import torch.nn as nn
import torch.optim as optim

import pp_pb2 as pp
from ppcore import modules as MODULES
from google.protobuf import text_format
from google.protobuf import pyext

import re
import os
import inspect
import collections
from typing import Sequence


class Model(nn.Module):
    def __init__(self, model_param=None, model_file='./pp_model.prototxt'):
        super().__init__()
        
        _model_param = pp.ModelParameter()

        if model_param is None:
            text_format.Merge(open(model_file, 'rb').read(), _model_param)
        else:
            _model_param.CopyFrom(model_param)

        self.model = self.parse(_model_param)
        self.model_param = _model_param
        
        
    def forward(self, data):
        
        outputs = {}
        outputs.update(data)

        for module, param in zip(self.model, self.model_param.module,):
            
            _outputs = module(*[outputs[b] for b in param.bottom])
            if not isinstance(_outputs, Sequence):
                _outputs = (_outputs, )
                
            outputs.update({t:_outputs[i] for i, t in enumerate(param.top) if len(param.top)})

        for k in outputs:
            print(k, outputs[k].shape)
            
        return outputs
    
    
    def parse(self, config):
        '''parse
        '''
        modules = nn.ModuleList()
        
        for i, m in enumerate(config.module):
            _name = m.name
            _module = self._parse_module(m, MODULES)
            
            modules.append( _module )
            
        return modules
    
    
    @staticmethod
    def _parse_module(config, classes):
        '''instantiate a module
        '''

        if config.type == 'Custom':
            _code = config.custom_param.module_inline if config.custom_param.module_inline \
                else open(config.custom_param.module_file, 'r').read()
            exec(_code)
            
            return locals()[config.name]
            
        _type = config.type
        _class = classes[_type]
        
        argspec = inspect.getfullargspec(_class.__init__)
        argsname = [arg for arg in argspec.args if arg != 'self']

        kwargs = {}

        if argspec.defaults is not None:
            kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

        if hasattr(config, f'{_type.lower()}_param'):
            _param = getattr(config, f'{_type.lower()}_param')
            _param = {k.name: v for k, v in _param.ListFields()}
            _param.update({k: (list(v)[0] if len(v) == 1 else list(v)) \
                          for k, v in _param.items() if isinstance(v, pyext._message.RepeatedScalarContainer)})
            
            kwargs.update({k: _param[k] for k in argsname if k in _param})

        kwargs = collections.OrderedDict([(k, kwargs[k]) for k in argsname])

        return _class(**kwargs)


# model = Model()
# model = Model()
# print(model)
# data = torch.rand(1, 20, 10, 10)
# model({'data': data})

print('---------')


class Solver(object):
    def __init__(self, solver_file='./pp_solver.prototxt'):
        
        solver_param = pp.SolverParameter() 
        text_format.Merge(open(solver_file, 'rb').read(), solver_param)
                
        if solver_param.model.ByteSize():
            model = Model(model_param=solver_param.model)
        else:
            model = Model(model_file=solver_param.model_file)
        
        if solver_param.optimizer.ByteSize():
            optimizer = self.parse_optimizer(solver_param.optimizer, model=model)
            
        self.model = model
        self.optimizer = optimizer
        self.dataloader = None
        
        
    def train(self, ):
        pass
    
    
    def test(self, ):
        pass
    
    
    def parse_optimizer(self, config, model):
        '''parse optimizer config
        '''
        # print(list(locals().keys()))
        
        assert config.type in dir(torch.optim), f'assert {config.type} exists'

        _class = getattr(torch.optim, config.type)
        
        argspec = inspect.getfullargspec(_class.__init__)
        argsname = [arg for arg in argspec.args if arg != 'self']

        kwargs = {}

        if argspec.defaults is not None:
            kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

        if len(config.params_group):
            params = []
            for group in config.params_group:
                exec(group.params_inline)
                _var_name = group.params_inline.split('=')[0].strip()                
                _params = {k.name:v for k, v in group.ListFields() if k.name != 'params_inline'}
                _params.update({'params': locals()[_var_name]})
                params.append(_params)
        else:
            params = model.parameters()
        
        kwargs.update( {'params': params})
        
        _param = {k.name: v for k, v in config.ListFields()}
        kwargs.update({k: _param[k] for k in argsname if k in _param})

        if config.code_file and False:
            with open(config.code_file, 'r') as f:
                exec( f.read() )
            
            assert 'optimizer' in locals().keys(), 'make sure var optimizer exists.'

            print( list(locals().keys()) )
            
            return locals()['optimizer'], locals()['lr_scheduler']

        print(config)
        
        return _class( **kwargs )


    
solver = Solver()

data = torch.rand(1, 20, 10, 10)
solver.model({'data': data})

