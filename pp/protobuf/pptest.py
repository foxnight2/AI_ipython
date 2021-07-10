import torch
import torch.nn as nn

import pp_pb2 as pp
from ppcore import modules as MODULES
from google.protobuf import text_format


import re
import inspect
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self, model_param=None, config='./pp_model.prototxt'):
        super().__init__()
        
        if model_param is None:
            model_param = pp.ModelParameter()
            text_format.Merge(open(config, 'rb').read(), model_param)
        
        print(model_param)
        self.model = self.parse(model_param)

        
    def forward(self, data):
        
        outputs = {}

    
    def parse(self, config):
        '''parse
        '''
        modules = nn.ModuleList()
        
        outputs = []
                
        for i, m in enumerate(config.module):
            
            assert m.type in MODULES, ''
            _class = MODULES[m.type]
            
            argspec = inspect.getfullargspec(_class.__init__)
            argsname = [arg for arg in argspec.args if arg != 'self']
            
            kwargs = {}
            if argspec.defaults is not None:
                kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

            _param = [name for name in dir(m) if '_param' in name.lower() and m.type.lower() in name.lower()]
            assert len(_param) == 1, f'must define {m.type.lower()}_param'
            
            param = {k.name: v for k, v in getattr(m, _param[0]).ListFields()}
            
            kwargs.update({k: param[k] for k in argsname if k in param})
            kwargs = OrderedDict([(k, kwargs[k]) for k in argsname])
            
            modules.append( _class(**kwargs) )
            
        return modules
    
    
# model = Model()
print('---------')


class Solver(object):
    def __init__(self, config='./pp_solver.prototxt'):
        
        solver_param = pp.SolverParameter()
        text_format.Merge(open(config, 'rb').read(), solver_param)
        
        model = Model(config=solver_param.model)
        
        print(model)
        print(solver_param)
        
    def train(self, ):
        pass
    
    
    def test(self, ):
        pass
    
    
    def parse(self, ):
        pass
    

    
solver = Solver()