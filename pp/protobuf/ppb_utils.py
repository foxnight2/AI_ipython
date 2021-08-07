import torch

import inspect

from collections import OrderedDict
from google.protobuf import pyext




def _build_module(clss, params):
    '''_build_module
    '''
    argspec = inspect.getfullargspec(clss.__init__)
    argsname = [arg for arg in argspec.args if arg != 'self']

    kwargs = {}
    if argspec.defaults is not None:
        kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )
        
    kwargs.update({k: params[k] for k in argsname if k in params})
    
    kwargs = OrderedDict([(k, kwargs[k]) for k in argsname])
    module = clss(**kwargs)
        
    return module


def _module_param(config,):
    '''get_module_param
    '''
    _type = config.type
    _name = [k.name for k, _ in config.ListFields() 
             if '_param' in k.name and f'{_type.lower()}param' == k.name.replace('_', '')]
    assert len(_name) <= 1, ''
    
    params = {}
    if len(_name) == 1:
        _param = getattr(config, _name[0])
        _param = {k.name: v for k, v in _param.ListFields()}
        
        for k, v in _param.items():
            if isinstance(v, pyext._message.RepeatedScalarContainer):
                v = list(v)
                v = v[0] if len(v) == 1 and k in ('kernel_size', 'stride', 'padding') else v
            
            params[k] = v
                
    return params


def build_module(config, modules):
    '''build protobuf module module/layer
    {
        name: "conv2d"
        type: Conv2d
        top: "data"
        conv2d_param {
            kernerl_size: 1
        }
    }
    ''' 
    if config.type == 'Custom':
        _code = config.custom_param.module_inline if config.custom_param.module_inline \
            else open(config.custom_param.module_file, 'r').read()
        exec(_code)

        return locals()[config.name]

    _param = _module_param(config)
    
    module = _build_module(modules[config.type], _param)
    
    module.top = list(config.top) if config.top else []
    module.bottom = list(config.bottom) if config.bottom else []
    
    return module


def build_optimizer(config, model, modules=torch.optim):
    '''optimizer
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )
        return locals()['optimizer']

    if len(config.params_group):
        params = []
        for group in config.params_group:
            exec(group.params_inline)
            _var_name = group.params_inline.split('=')[0].strip()                
            _params = {k.name: v for k, v in group.ListFields() if k.name != 'params_inline'}
            _params.update({'params': locals()[_var_name]})
            params.append(_params)
    else:
        params = model.parameters()
        
    _param = {k.name: v for k, v in config.ListFields()}
    _param.update( {'params': params})
    
    clss = getattr(modules, config.type)

    optimizer = _build_module(clss, _param) 

    return optimizer


def build_lr_scheduler(config, optimizer, modules=torch.optim.lr_scheduler):
    '''lr_scheduler
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )            
        return locals()['lr_scheduler']

    _param = {k.name: v for k, v in config.ListFields()}
    _param.update( {'optimizer': optimizer} )

    clss = getattr(modules, config.type)
    lr_scheduler = _build_module(clss, _param)

    return lr_scheduler


def build_dataloader(config, dataset, modules={'DataLoader': torch.utils.data.DataLoader}):
    '''parse dataloader
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )
        return locals()['dataloader']

    _param = {k.name: v for k, v in config.ListFields()}
    _param.update( {'dataset': dataset} )     

    dataloader = _build_module(modules[config.type], _param)

    dataloader.shuffle = _param['shuffle'] if 'shuffle' in _param else False
    
    return dataloader
    

    

if __name__ == '__main__':
    
    
    pass


"""

def camel_to_underscore(name, suffix='param'):
    '''DummyDataset -> dummy_dataset_param
    '''
    _splits = re.findall('[A-Z][^A-Z]*', name)
    _splits = [_s.lower() for _s in _splits]
    
    if suffix is not None:
        _splits.append(suffix)
    
    return '_'.join(_splits)


def get_module_param(_type, config,):
    '''get_module_param
    '''
    _param_name = [k.name for k, _ in config.ListFields() 
                   if '_param' in k.name and f'{_type.lower()}param' == k.name.replace('_', '')]
    assert len(_param_name) <= 1, ''
    
    params = {}
    
    if len(_param_name) == 1:
        return getattr(config, _param_name[0])
    
    return params


def build_module(config):
    '''
    {
        type: Conv2d
        top: data
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


    
# parse module
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

    _param_name = [k.name for k, _ in config.ListFields() 
                   if '_param' in k.name and f'{_type.lower()}param' == k.name.replace('_', '')]
    assert len(_param_name) <= 1, ''
    
    # if hasattr(config, f'{_type.lower()}_param'):
    if len(_param_name):
        # _param = getattr(config, f'{_type.lower()}_param')
        _param = getattr(config, _param_name[0])
        _param = {k.name: v for k, v in _param.ListFields()}
        
        _param.update({k: (list(v)[0] if len(v) == 1 else list(v)) \
                      for k, v in _param.items() if isinstance(v, pyext._message.RepeatedScalarContainer)})

        kwargs.update({k: _param[k] for k in argsname if k in _param})

    kwargs = collections.OrderedDict([(k, kwargs[k]) for k in argsname])
    module = _class(**kwargs)

    if config.reset_inline or config.reset_file:
        _code = config.reset_inline if config.reset_inline \
                                    else open(config.reset_file, 'r').read()
        exec( _code )

    return locals()['module']
    
    

# parse optimizer
def _parse_optimizer(config, model, classes=torch.optim):
    '''parse optimizer config
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )            
        return locals()['optimizer']

    _class = getattr(classes, config.type)

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


    
# parse lr scheduler
def _parse_lr_scheduler(config, optimizer, classes=torch.optim.lr_scheduler):
    '''parse lr_scheduler config
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )            
        return locals()['lr_scheduler']

    _class = getattr(classes, config.type)

    argspec = inspect.getfullargspec(_class.__init__)
    argsname = [arg for arg in argspec.args if arg != 'self']

    kwargs = {}

    if argspec.defaults is not None:
        kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

    _param = {k.name: v for k, v in config.ListFields()}
    kwargs.update({k: _param[k] for k in argsname if k in _param})

    kwargs.update( {'optimizer': optimizer} )

    _lr_scheduler = _class( **kwargs )

    return _lr_scheduler    



# parse dataloader
def _parse_dataloader(config, dataset, ):
    '''parse dataloader
    '''
    if config.module_file or config.module_inline:
        _code = config.module_inline if config.module_inline else open(config.module_file, 'r').read()
        exec( _code )            
        return locals()['dataloader']

    _class = torch.utils.data.DataLoader

    argspec = inspect.getfullargspec(_class.__init__)
    argsname = [arg for arg in argspec.args if arg != 'self']

    kwargs = {}

    if argspec.defaults is not None:
        kwargs.update( dict(zip(argsname[::-1], argspec.defaults[::-1])) )

    _param = {k.name: v for k, v in config.ListFields()}
    kwargs.update({k: _param[k] for k in argsname if k in _param})

    kwargs.update( {'dataset': dataset} )        
    
    _dataloader = _class( **kwargs )

    _dataloader.shuffle = _param['shuffle'] if 'shuffle' in _param else False
    
    return _dataloader
    
"""