import torch
import torch.nn as nn
import torch.optim as optim

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

    # @torch.cuda.amp.autocast(enabled=False)
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
        
        solver = pp.SolverParameter() 
        text_format.Merge(open(solver_file, 'rb').read(), solver)        
        
        model = Model(solver.model, solver.model_file)
                
        # reader = pp.SolverParameter() 
        # text_format.Merge(open(solver.reader_file, 'rb').read(), reader)
        
        # solver_str = text_format.MessageToString(solver, indent=2)
        # reader_str = text_format.MessageToString(reader, indent=2)

        
        if solver.optimizer.ByteSize():
            optimizer = utils.build_optimizer(solver.optimizer, model)
            
        if solver.lr_scheduler.ByteSize():
            lr_scheduler = utils.build_lr_scheduler(solver.lr_scheduler, optimizer)

        # TODO merge in one
        if len(solver.transforms):
            transforms = {_m.name: utils.build_transforms(_m, MODULES) for _m in solver.transforms}
            
        if len(solver.dataset):
            dataset = {_m.name: utils.build_dataset(_m, transforms, MODULES) for _m in solver.dataset}
            
        if len(solver.dataloader):
            dataloader = {_m.phase: utils.build_dataloader(_m, dataset[_m.dataset]) for _m in solver.dataloader}

        if solver.distributed.ByteSize() and solver.distributed.enabled: 
            device, model, dataloader = utils.setup_distributed(solver.distributed, 
                                                                model, 
                                                                (dataloader if dataloader else None))
        else:
            device = torch.device(solver.device)
            model = model.to(device)
            
        [setattr(self, k, m) for k, m in locals().items() if k in ['model', 'device', 'dataloader', 'optimizer', 'lr_scheduler']]
        
        # self.model = model
        # self.device = device
        # self.optimizer = optimizer
        # self.lr_scheduler = lr_scheduler
        # self.dataloader = dataloader
        
        self.last_epoch = 0
        self.epoches = solver.epoches
        
        self.use_amp = solver.use_amp
        
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html?highlight=scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.clip_grad_norm = 1
        
        if solver.resume:
            self.restore(solver.resume)
            
        # print(reader_str)
        # print(solver_str)
        # print('xxxxxxxxxxxx')
        # print(reader.dataset)
        # print(solver.dataset)
        
        
    def train(self, ):
        self.model.train()
        dataloader = self.dataloader[1] # phase 1 - train
        scaler = self.scaler
        enabled_clip = self.clip_grad_norm > 0
        
        for e in range(self.last_epoch, self.epoches):
            
            utils.start_timer()
            
            if hasattr(self.model, 'distributed') and self.model.distributed:
                dataloader.sampler.set_epoch(e)
                
            for _, blob in enumerate(dataloader):
                                
                if not isinstance(blob, dict):
                    blob = blob if isinstance(blob, Sequence) else (blob, )
                    blob = {n: d for n, d in zip(dataloader.dataset.top, blob)}

                blob.update({k: t.to(self.device) for k, t in blob.items() if isinstance(t, torch.Tensor)})

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                
                    outputs = self.model(blob)
                    print([(k, v.dtype) for k, v in outputs.items()])
                    
                    loss = outputs['loss'].mean()
                    print('loss', loss.item())
                    
                    scaler.scale(loss).backward()
                    
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.)
                    
                    # scaler.step() first unscales the gradients of the optimizer's assigned params.
                    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(self.optimizer)
                    scaler.update()

                self.optimizer.zero_grad()

            self.lr_scheduler.step()
            
            utils.end_timer_and_print(f'use_amp: {self.use_amp}')
            
            print('--------')
        
        self.save(str(e))
                
    
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
        model = self.model.module if utils.is_dist_available_and_initialized() else self.model
        state = {
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'last_epoch': self.last_epoch,
        }
        utils.save_on_main(state,  prefix + '.pt')
            
        
    def restore(self, path):
        '''restore
        '''       
        # consume_prefix_in_state_dict_if_present()
        # map_location

        state = torch.load(path, map_location='cpu')

        if utils.is_dist_available_and_initialized():
            self.model.module.load_state_dict(state['model'])
        else:
            self.model.load_state_dict(state['model'])
            
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        self.last_epoch = state['last_epoch']
    

        
if __name__ == '__main__':
    
    # python -m torch.distributed.launch --nproc_per_node=2 pptest.py -c pp_cls.prototxt
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, ) # torch.distributed.launch
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('-c', '--config', )

    args = parser.parse_args()

    solver = Solver(args.config)
    solver.train()
    solver.test()
    
    
    # ------- onnx
    
#     if utils.get_rank() == 0:
        
#         model = solver.model
#         model.eval()

#         data = torch.randn(10, 3, 224, 224).cuda()
#         outputs = model(data)


#         input_names = ['data']
#         output_names = [n for n in outputs]

#         print([(k, v.dtype, v.device) for k, v in outputs.items()])
#         print([v.cpu().data.numpy().sum() for _, v in outputs.items()])

#         torch.onnx.export(model, 
#                           data, 
#                           'test.onnx', 
#                           verbose=False, 
#                           export_params=True,
#                           input_names=input_names, 
#                           output_names=output_names, 
#                           opset_version=10, 
#                           dynamic_axes={'data': {0: 'batch_size'}, }, )

#         import onnx
#         import onnxruntime

#         onnx_model = onnx.load('test.onnx')
#         onnx.checker.check_model(onnx_model)

#         ort_session = onnxruntime.InferenceSession("test.onnx")
#         print([d.name for d in ort_session.get_inputs()])

#         # data = torch.randn(4, 3, 224, 224, device='cuda')

#         ort_outs = ort_session.run(None, {'data': data.cpu().numpy()})
#         print([out.sum() for out in ort_outs])
#         print([out.shape for out in ort_outs])