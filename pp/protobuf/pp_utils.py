

from collections import OrderedDict
from typing import Dict, Any

from google.protobuf import pyext




def get_module_param(config, with_type=True):
    '''get_module_param
    '''
    _name = [k.name for k, _ in config.ListFields() 
             if '_param' in k.name and f'{_type.lower()}param' == k.name.replace('_', '')]
    assert len(_name) <= 1, ''
    
    params = {}
    if len(_name) == 1 and use_type:
        _param = getattr(config, _name[0])
        _param = {k.name: v for k, v in _param.ListFields()}
        
        for k, v in _param.items():
            if isinstance(v, pyext._message.RepeatedScalarContainer):
                v = list(v)
                v = v[0] if len(v) == 1 and k in ('kernel_size', 'stride', 'padding') else v
            
            params[k] = v
                
    return params
        
    
def build_module(config, modules: Dict[str, Any]):
    '''build protobuf module
    {
        name: "conv2d"
        type: Conv2d
        top: "data"
        conv2d_param {
            kernerl_size: 1
            
            filter {
                type: Constant
                value: 0.1
            }
            filter {
                type: Constant
                value: 0.1
            }
        }
    }
    '''
    _type = config.type
    _clss = modules[_type]
    
    argspec = inspect.getfullargspec(_class.__init__)
    argsname = [arg for arg in argspec.args if arg != 'self']

    kwargs = {}
    if argspec.defaults is not None:
        kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

    kwargs.update(get_module_param(config))
    
    kwargs = OrderedDict([(k, kwargs[k]) for k in argsname])
    module = _class(**kwargs)
    
    config['module'] = module
    
    return module



if __name__ == '__main__':
    
    
    