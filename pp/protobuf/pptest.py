import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, SequentialSampler
# from torch.utils.tensorboard import SummaryWriter

from ppcore import modules as MODULES

import pp_pb2 as pp
from google.protobuf import text_format
from google.protobuf import pyext

import re
import os
import inspect
import collections
from typing import Sequence
from tqdm import tqdm


import ppb_utils as utils



    
# distributed
def setup_distributed(config, model=None, dataloader=None):
    '''distributed setup
    reference: https://github.com/facebookresearch/detr/blob/master/util/misc.py#L406
    '''
    dist.init_process_group(backend = (config.backend if config.backend else 'nccl'),
                            init_method = (config.init_method if config.init_method else 'env://' ), 
                            # world_size = (config.world_size if config.init_method else args.world_size), 
                            # rank = args.local_rank 
                           )

    device = torch.device(f'cuda:{args.local_rank}')

    torch.cuda.set_device(device)
    torch.distributed.barrier()
    distributed_print(args.local_rank == 0)

    # model
    if model is not None:
        model.to(device)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if dataloader is not None:         
        _sampler = {k: DistributedSampler(v.dataset, shuffle=v.shuffle) for k, v in dataloader.items()}
        dataloader = {k: DataLoader(v.dataset, 
                                    v.batch_size, 
                                    sampler=_sampler[k], 
                                    drop_last=v.drop_last, 
                                    collate_fn=v.collate_fn, 
                                    num_workers=v.num_workers) for k, v in dataloader.items()}

    return device, model, dataloader
    
    
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
                
        # outputs = collections.defaultdict(lambda:None)
        outputs = {}
        outputs.update(data)

        # for module, param in zip(self.model, self.model_param.module,):
        for module in self.model:
            
            if self.training and not (module.phase == 0 or module.phase == 1):
                continue
            if not self.training and not (module.phase == 0 or module.phase == 2):
                continue

            _outputs = module(*[outputs[b] for b in module.bottom])
            if not isinstance(_outputs, Sequence):
                _outputs = (_outputs, )

            outputs.update({t: _outputs[i] for i, t in enumerate(module.top) if len(module.top)})

        return outputs
    
    
    def parse(self, config):
        '''parse
        '''
        modules = nn.ModuleList()
        
        train_modules = nn.ModuleList()
        eval_modules = nn.ModuleList()
        
        for i, m in enumerate(config.module):
            
            _module = utils.build_module(m, MODULES)
            
            modules.append( _module )
            
            if m.phase == 0 or m.phase == 1:
                train_modules.append(_module)
            elif m.phase == 0 or m.phase == 2:
                eval_modules.append(_module)

        print(train_modules)
        print(eval_modules)
        
        return modules
    
    
    def export_onnx(self, ):
        pass
    
    

class Solver(object):
    def __init__(self, solver_file='./pp_solver.prototxt'):
        
        solver_param = pp.SolverParameter() 
        text_format.Merge(open(solver_file, 'rb').read(), solver_param)        
        
        model = Model(solver_param.model, solver_param.model_file)
            
        if solver_param.optimizer.ByteSize():
            # optimizer = _parse_optimizer(solver_param.optimizer, model)
            optimizer = utils.build_optimizer(solver_param.optimizer, model)
            
        if solver_param.lr_scheduler.ByteSize():
            # lr_scheduler = _parse_lr_scheduler(solver_param.lr_scheduler, optimizer)
            lr_scheduler = utils.build_lr_scheduler(solver_param.lr_scheduler, optimizer)

        if len(solver_param.dataset):
            # dataset = {_m.name: _parse_module(_m, MODULES) for _m in solver_param.dataset}
            dataset = {_m.name: utils.build_module(_m, MODULES) for _m in solver_param.dataset}
            dataset_tops = {_m.name: _m.top for _m in solver_param.dataset}
            
        if len(solver_param.dataloader):
            # dataloader = {_m.name: _parse_dataloader(_m, dataset[_m.dataset]) for _m in solver_param.dataloader}
            dataloader = {_m.phase: utils.build_dataloader(_m, dataset[_m.dataset]) for _m in solver_param.dataloader}

        if solver_param.distributed.ByteSize():
            device, model, dataloader = setup_distributed(solver_param.distributed, model, (dataloader if dataloader else None))
        else:
            device = torch.device(solver_param.device)
            model = model.to(device)
        
        [setattr(self, k, m) for k, m in locals().items() if k in ['model', 'device', 'dataloader', 'optimizer', 'lr_scheduler']]
        
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        # self.dataloader = dataloader
        
        self.last_epoch = 0
        self.epoches = solver_param.epoches
        
        print(solver_param)

        
    def train(self, ):
        self.model.train()
        dataloader = self.dataloader[1] # phase 1 - train
        
        data = torch.rand(1, 3, 10, 10).to(self.device)
        
        for e in range(self.last_epoch, self.epoches):
            if hasattr(self, 'distributed') and self.distributed is True:
                dataloader.sampler.set_epoch(e)
                
            for _, blob in enumerate(dataloader):
                
                print(type(blob), len(blob))

                if not isinstance(blob, dict):
                    blob = blob if isinstance(blob, Sequence) else (blob, )
                    blob = {n: d for n, d in zip(dataloader.dataset.top, blob)}

                blob.update({k: t.to(self.device) for k, t in blob.items() if isinstance(t, torch.Tensor)})
                
                output = self.model(blob)
            
            print('--------')
                
    
    def test(self, ):
        self.model.eval()
        dataloader = self.dataloader[2] # phase 1 - eval

        for _, blob in enumerate(dataloader):

            print(type(blob), len(blob))

            if not isinstance(blob, dict):
                blob = blob if isinstance(blob, Sequence) else (blob, )
                blob = {n: d for n, d in zip(dataloader.dataset.top, blob)}

            blob.update({k: t.to(self.device) for k, t in blob.items() if isinstance(t, torch.Tensor)})

            output = self.model(blob)

            for k, v in output.items():
                print(k, (v.shape if v is not None else None))

        print('--------')

    
    
    
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
    
    


if __name__ == '__main__':
    
    # python -m torch.distributed.launch --nproc_per_node=2 pptest.py

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, ) # torch.distributed.launch
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    args = parser.parse_args()

    solver = Solver()
    solver.train()

    solver.test()