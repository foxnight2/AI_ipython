

from collections import OrderedDict
from typing import Dict, Any




def get_module_param(config):
    _name = [k.name for k, _ in config.ListFields() 
             if '_param' in k.name and f'{_type.lower()}param' == k.name.replace('_', '')]
    assert len(_name) <= 1, ''
    
    params = {}
    if len(_name) == 1:
        _param = getattr(config, _name[0])
        
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

    # {k: v for k, v in get_module_param(config).items() if k in ('kernel_size', 'stride', 'padding', 'dilation') and len(v)==1}
    
    kwargs.update()
    
    kwargs = OrderedDict([(k, kwargs[k]) for k in argsname])
    module = _class(**kwargs)
    
    config['module'] = module
    
    return module