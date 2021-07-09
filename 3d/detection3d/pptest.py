import torch
import torch.nn as nn

import pp_pb2 as pp
from ppcore import modules as MODULES
from google.protobuf import text_format


import re
import inspect
from collections import defaultdict, OrderedDict


class Model(nn.Module):
    def __init__(self, config='./pp.prototxt'):
        super().__init__()
        model_params = pp.ModelParameter()
        text_format.Merge(open(config, 'rb').read(), model_params)
        
            
        self.model_params = model_params
        self.model = self.parse(model_params)
        
    def forward(self, x):
        pass
    
        
    def parse(self, config):
        '''parse
        '''
        modules = []
        for i, m in enumerate(config.module):
            assert m.type in MODULES, ''
            _class = MODULES[m.type]
            
            argspec = inspect.getfullargspec(_class.__init__)
            argsname = [arg for arg in argspec.args if arg != 'self']
            
            kwargs = {}
            if argspec.defaults is not None:
                kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

            _param = [name for name in dir(m) if '_param' in name.lower() and m.type.lower() in name.lower()]
            assert len(_param) == 1, ''
            param = getattr(m, _param[0])
            param = {k.name: v for k, v in param.ListFields()}
            
            kwargs.update({k: param[k] for k in argsname if k in param})
            
            kwargs = OrderedDict([(k, kwargs[k]) for k in argsname])
            
            modules.append( _class(**kwargs) )
            
        return modules
    
    
model = Model()
print(model)