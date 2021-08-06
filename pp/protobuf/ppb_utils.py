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