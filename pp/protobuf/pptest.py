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

        if model_param is not None and model_param.ByteSize():
            _model_param.CopyFrom(model_param) 
        else:
            text_format.Merge(open(model_file, 'rb').read(), _model_param)

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

        model = Model(solver_param.model, solver_param.model_file)
        # self.model = model
        
        if solver_param.optimizer.ByteSize():
            optimizer = self.parse_optimizer(solver_param.optimizer, model=model)
            # optimizer = self.parse(solver_param.optimizer, torch.optim)
            # self.optimizer = optimizer
            
        if solver_param.lr_scheduler.ByteSize():
            lr_scheduler = self.parse_lr_scheduler(solver_param.lr_scheduler, optimizer)
            # lr_scheduler = self.parse(solver_param.lr_scheduler, torch.optim.lr_scheduler)
            # self.lr_scheduler = lr_scheduler
        
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = None
        
        print(model, optimizer, lr_scheduler)
        
    def train(self, ):
        pass
    
    
    def test(self, ):
        pass
    
    
    def parse(self, config, classes):
        '''parse optimizer lr_scheduler 
        '''
        if config.module_file or config.module_inline:
            _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
            exec( _code )            
            return locals()[config.name]
        
        _class = getattr(classes, config.type)
        
        argspec = inspect.getfullargspec(_class.__init__)
        argsname = [arg for arg in argspec.args if arg != 'self']

        kwargs = {}

        if argspec.defaults is not None:
            kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )
        
        _param = {k.name: v for k, v in config.ListFields()}
        
        if 'params' in argsname:
            if 'params_group' in _param and len(_param['params_group']):
                params = []
                for group in config.params_group:
                    exec(group.params_inline)
                    _var_name = group.params_inline.split('=')[0].strip()                
                    _params = {k.name:v for k, v in group.ListFields() if k.name != 'params_inline'}
                    _params.update({'params': locals()[_var_name]})
                    params.append(_params)
            else:
                params = self.model.parameters()
            
            kwargs.update( {'params': params})

        elif 'optimizer' in argsname:
            
            kwargs.update( {'optimizer': self.optimizer} )

        else:
            pass
        
        kwargs.update({k: _param[k] for k in argsname if k in _param})
        
        return _class( **kwargs )
    

    @staticmethod
    def parse_optimizer(config, model):
        '''parse optimizer config
        '''
        if config.module_file or config.module_inline:
            _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
            exec( _code )            
            return locals()['optimizer']
        
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
        
        _optimizer = _class( **kwargs )
        
        return _optimizer

    
    @staticmethod
    def parse_lr_scheduler(config, optimizer):
        '''parse lr_scheduler config
        '''
        if config.module_file or config.module_inline:
            _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
            exec( _code )            
            return locals()['lr_scheduler']
        
        _class = getattr(torch.optim.lr_scheduler, config.type)
        
        argspec = inspect.getfullargspec(_class.__init__)
        argsname = [arg for arg in argspec.args if arg != 'self']

        kwargs = {}

        if argspec.defaults is not None:
            kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

        kwargs.update( {'optimizer': optimizer} )
        
        _param = {k.name: v for k, v in config.ListFields()}
        kwargs.update({k: _param[k] for k in argsname if k in _param})
        
        _lr_scheduler = _class( **kwargs )
        
        return _lr_scheduler    

    
solver = Solver()

data = torch.rand(1, 20, 10, 10)
solver.model({'data': data})

