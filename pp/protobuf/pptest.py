import torch
import torch.nn as nn

import pp_pb2 as pp
from ppcore import modules as MODULES
from google.protobuf import text_format
from google.protobuf import pyext

import re
import inspect
import collections
from typing import Sequence

class Model(nn.Module):
    def __init__(self, model_param=None, config='./pp_model.prototxt'):
        super().__init__()
        
        if model_param is None:
            model_param = pp.ModelParameter()
            text_format.Merge(open(config, 'rb').read(), model_param)
        
        print(model_param)
        self.model = self.parse(model_param)
        self.model_param = model_param
        
        
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
            
            assert m.type in MODULES, ''
            _name = m.name
            _type = m.type
            _class = MODULES[_type]

            argspec = inspect.getfullargspec(_class.__init__)
            argsname = [arg for arg in argspec.args if arg != 'self']
            
            kwargs = {}
            if argspec.defaults is not None:
                kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )
            
            if hasattr(m, f'{_type.lower()}_param'):
                _param = getattr(m, f'{_type.lower()}_param')
                _param = {k.name: v for k, v in _param.ListFields()}
                _param.update({k: (list(v)[0] if len(v) == 1 else list(v)) \
                              for k, v in _param.items() if isinstance(v, pyext._message.RepeatedScalarContainer)})
                
                kwargs.update({k: _param[k] for k in argsname if k in _param})
            
            kwargs = collections.OrderedDict([(k, kwargs[k]) for k in argsname])
            modules.append( _class(**kwargs) )
            
        return modules
    
    
# model = Model()
print('---------')


class Solver(object):
    def __init__(self, config='./pp_solver.prototxt'):
        
        solver_param = pp.SolverParameter()
        text_format.Merge(open(config, 'rb').read(), solver_param)
        
        model = Model(config=solver_param.model)
        # MergeFrom
        
        self.model = model
        
        print(model)
        print(solver_param)
        
    def train(self, ):
        pass
    
    
    def test(self, ):
        pass
    
    
    def parse(self, ):
        pass
    

    
# solver = Solver()

model = Model()
print(model)
data = torch.rand(1, 20, 10, 10)
model({'data': data})

