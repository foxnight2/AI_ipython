import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, SequentialSampler


from ppcore import modules as MODULES

import pp_pb2 as pp
from google.protobuf import text_format
from google.protobuf import pyext

import re
import os
import inspect
import collections
from collections import OrderedDict
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
    setup_distributed_print(args.local_rank == 0)

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
    
    
def setup_distributed_print(is_master):
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

        self.input_names = list(_model_param.input_names)
        self.output_names = list(_model_param.output_names)

        self.model = self.parse(_model_param)
        self.model_param = _model_param

         
    def forward(self, data):
        '''
        data (dict | Tensor | Tuple[tensor])
        '''
        # outputs = collections.defaultdict(lambda:None)
        outputs = OrderedDict()
        
        if not isinstance(data, dict):
            assert len(self.input_names) > 0, f'{self.input_names}'
            data = data if isinstance(data, Sequence) else (data, )
            data = {k: v for k, v in zip(self.input_names, data)}
            
        outputs.update(data)

        inputs_id = [id(v) for _, v in outputs.items()] 
        
        for module in self.model:
            
            if self.training and not (module.phase == 0 or module.phase == 1):
                continue
            if not self.training and not (module.phase == 0 or module.phase == 2):
                continue

            _outputs = module(*[outputs[b] for b in module.bottom])
            _outputs = _outputs if isinstance(_outputs, Sequence) else (_outputs, )

            outputs.update({t: _outputs[i] for i, t in enumerate(module.top) if len(module.top)})
        
        # remove inputs in outputs for onnx purpose
        outputs = OrderedDict([(k, v) for k, v in outputs.items() if id(v) not in inputs_id])
        
        if self.output_names:
            outputs = OrderedDict([(n, outputs[n]) for n in self.output_names if n in outputs])
            
        return outputs
    
    
    def parse(self, config):
        '''parse
        '''
        modules = nn.ModuleList()
                
        for i, m in enumerate(config.module):
            
            _module = utils.build_module(m, MODULES)
            
            modules.append( _module )
            
        
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
    
    print(solver.model)
    
    
    
    
    solver.model.module.eval()
    
    data = torch.randn(10, 3, 224, 224, device='cuda')
    outputs = solver.model.module(data)
    
    input_names = ['data']
    output_names = [n for n in outputs]

    print([k for k, _ in outputs.items()])
    print([v.cpu().data.numpy().sum() for _, v in outputs.items()])
    
    torch.onnx.export(solver.model.module, 
                      data, 
                      'test.onnx', 
                      verbose=True, 
                      export_params=True,
                      input_names=input_names, 
                      output_names=output_names, 
                      opset_version=10, 
                      dynamic_axes={'data': {0: 'batch_size'}, }, )
    
    import onnx
    import onnxruntime

    onnx_model = onnx.load('test.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("test.onnx")
    print([d.name for d in ort_session.get_inputs()])
    
    # data = torch.randn(4, 3, 224, 224, device='cuda')

    ort_outs = ort_session.run(None, {'data': data.cpu().numpy()})
    print([out.sum() for out in ort_outs])
    print([out.shape for out in ort_outs])