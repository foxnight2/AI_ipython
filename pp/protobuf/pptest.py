import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import pp_pb2 as pp
from ppcore import modules as MODULES
from google.protobuf import text_format
from google.protobuf import pyext

import re
import os
import inspect
import collections
from typing import Sequence





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

    if hasattr(config, f'{_type.lower()}_param'):
        _param = getattr(config, f'{_type.lower()}_param')
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
    
    
# ---
def distributed_print(is_master):
    """
    reference: https://github.com/facebookresearch/detr/blob/master/util/misc.py
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
    
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
            
        return outputs
    
    
    def parse(self, config):
        '''parse
        '''
        modules = nn.ModuleList()
        
        for i, m in enumerate(config.module):
            _name = m.name
            _module = _parse_module(m, MODULES)

            modules.append( _module )
            
        return modules
    
    


# model = Model()
# model = Model()
# print(model)
# data = torch.rand(1, 20, 10, 10)
# model({'data': data})


class Solver(object):
    def __init__(self, solver_file='./pp_solver.prototxt'):
        
        solver_param = pp.SolverParameter() 
        text_format.Merge(open(solver_file, 'rb').read(), solver_param)

        model = Model(solver_param.model, solver_param.model_file)
        self.model = model
        
        if solver_param.optimizer.ByteSize():
            optimizer = _parse_optimizer(solver_param.optimizer, model)
            self.optimizer = optimizer
            
        if solver_param.lr_scheduler.ByteSize():
            lr_scheduler = _parse_lr_scheduler(solver_param.lr_scheduler, optimizer)
            self.lr_scheduler = lr_scheduler
            
        if len(solver_param.dataset):
            dataset = {_m.name: _parse_module(_m, MODULES) for _m in solver_param.dataset}
            self.dataset_tops = {_m.name: _m.top for _m in solver_param.dataset}
            
        if len(solver_param.dataloader):
            dataloader = {_m.name: _parse_dataloader(_m, dataset[_m.dataset]) for _m in solver_param.dataloader}
            self.dataloader = dataloader
        
        if solver_param.distributed.ByteSize():
            self.setup_distributed(solver_param.distributed)
        else:
            self.device = torch.device(solver_param.device)
            self.model = self.model.to(self.device)
        
        self.last_epoch = 0
        self.epoches = solver_param.epoches
        
        
    def train(self, ):
        self.model.train()

        data = torch.rand(1, 3, 10, 10).to(self.device)
        
        for e in range(self.last_epoch, self.epoches):
            if hasattr(self, 'distributed') and self.distributed is True:
                self.dataloader['train_dataloader'].sampler.set_epoch(e)
                
            for _, blob in enumerate(self.dataloader['train_dataloader']):
                
                print(blob.sum())

                if not isinstance(blob, dict):
                    blob = blob if isinstance(blob, Sequence) else (blob, )
                    blob = {n: d for n, d in zip(self.dataset_tops['train_dataset'], blob)}
                    
                blob.update({k: t.to(self.device) for k, t in blob.items() if isinstance(t, torch.Tensor)})
                
                output = self.model(blob)
            
            print('--------')
            
    def test(self, ):
        pass
    

    def save(self, prefix=''):
        '''save state
        '''
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'last_epoch': self.last_epoch
        }
        torch.save(state, prefix + '.pt')


    def restore(self, path):
        '''restore
        '''
        state = torch.load(path)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.model.load_state_dict(state['lr_scheduler'])
        self.last_epoch = state['last_epoch']
    
    

    def setup_distributed(self, config):
        '''distributed setup
        reference: https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406
        '''
        dist.init_process_group(backend = (config.backend if config.backend else 'nccl'),
                                init_method = (config.init_method if config.init_method else 'env://' ), 
                                # world_size = (config.world_size if config.init_method else args.world_size), 
                                # rank = args.local_rank 
                               )
        
        self.distributed = True
        self.device = torch.device(f'cuda:{args.local_rank}')
        
        torch.cuda.set_device(self.device)
        torch.distributed.barrier()

        # model
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[args.local_rank], output_device=args.local_rank)


        # dataloader            
        _sampler = {k: DistributedSampler(v.dataset, shuffle=v.shuffle) for k, v in self.dataloader.items()}
        _dataloader = {k: DataLoader(v.dataset, 
                                     v.batch_size, 
                                     sampler=_sampler[k], 
                                     drop_last=v.drop_last, 
                                     collate_fn=v.collate_fn, 
                                     num_workers=v.num_workers) for k, v in self.dataloader.items()}
        
        self.dataloader = _dataloader
        
        distributed_print(args.local_rank == 0)

                
if __name__ == '__main__':
    
    # python -m torch.distributed.launch --nproc_per_node=2 pptest.py

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, ) # torch.distributed.launch
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    args = parser.parse_args()

    solver = Solver()
    solver.train()

    