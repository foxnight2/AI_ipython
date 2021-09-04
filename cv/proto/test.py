import torch
import torchvision

import cv_pb2 as cvpb
from google.protobuf import text_format
from google.protobuf import json_format
# from google.protobuf import pyext

import inspect
from collections import OrderedDict


solver = cvpb.Solver()
text_format.Merge(open('./sovler.prototxt', 'rb').read(), solver)
# print(solver)

# WhichOneof
# print(solver.model.module[0].WhichOneof('param'))
# print(getattr(solver.model.module[0], solver.model.module[0].WhichOneof('param')))
# c+=1


# _param = {k.name: v for k, v in solver.transforms.op[1].ListFields()}
# print(_param)


# print(solver)
# print(type(solver.transforms.op[0]))
# op = solver.transforms.op[0]
# print(dir(op))
# print(op.ListFields())
# print(op.name)
# print(op.type)
# print(op.keep_ratio)
# print(op.size)


def build_module(clss, params):
    '''build_module
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


class ToTensor():
    def __init__(self) -> None:
        pass

class Resize():
    def __init__(self, size, keep_ratio) -> None:
        self.size = size
        self.keep_ration = keep_ratio


class Test(torch.nn.Module):
    def __init__(self, m) -> None:
        super().__init__()

        self.m = m

    def __repr__(self) -> str:
        return f'Test_{(self.m)}, {id(self)}'


class CocoDet(torch.utils.data.Dataset):
    def __init__(self, path, transforms) -> None:
        super().__init__()
        self.path = path
        self.transforms = transforms
        # print(transforms)
        # print(path)

    def __len__(self, ):
        return 10

    def __getitem__(self, ):
        pass


class Compose(torchvision.transforms.Compose):
    def __init__(self, op):
        super().__init__(op)
        

class Mosaic(object):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
    
    def __call__(self, img):
        return img 


modules = {m: getattr(torch.nn, m) for m in dir(torch.nn) \
    if inspect.isclass(getattr(torch.nn, m)) and issubclass(getattr(torch.nn, m), torch.nn.Module)}

modules.update({m: getattr(torch.optim, m) for m in dir(torch.optim) \
    if inspect.isclass(getattr(torch.optim, m)) and issubclass(getattr(torch.optim, m), torch.optim.Optimizer)})

modules.update({m: getattr(torch.optim.lr_scheduler, m) for m in dir(torch.optim.lr_scheduler) \
    if inspect.isclass(getattr(torch.optim.lr_scheduler, m)) })

modules.update({
    'Test': Test,
    'ToTensor': ToTensor,  
    'Resize': Resize,  
    'Conv2d': torch.nn.Conv2d,
    'ReLU': torch.nn.ReLU,
    'CocoDet': CocoDet,
    'DataLoader': torch.utils.data.DataLoader,
    'Compose': Compose,
    'Mosaic': Mosaic,
})


solver_dict = json_format.MessageToDict(solver, preserving_proto_field_name=True, including_default_value_fields=False)
# print(solver_dict)



def merge():
    '''merge
    '''
    pass


def build(config, mm):
    '''build
    mm cache all build modules.
    '''
    if not isinstance(config, (dict, list)):
        return

    for i, k in enumerate(config):
        v = config[k] if isinstance(config, dict) else k
        m = build(v, mm)

        if m is not None:
            assert not (hasattr(m, 'top') or hasattr(m, 'bottom')), f'{m} .top, .bottom'
            if 'top' in v or 'bottom' in v:
                m.top = v.get('top', None)
                m.bottom = v.get('bottom', None)

            config[k if isinstance(config, dict) else i] = m
            
            # mm.update({v['name']: m} if 'name' in v else {})
            if 'name' in v:
                assert v['name'] not in mm, f"name {v['name']} already exists."
                mm.update({v['name']: m})


    if isinstance(config, dict) and 'type' in config:
        k = [k for k in config if '_param' in k]
        v = config[k[0]] if len(k) != 0 else {}
        v.update({_k: mm[_v] for _k, _v in v.items() if isinstance(_v, str) and _v in mm})

        m = build_module(modules[config['type']], v)

        return m

    # model
    elif isinstance(config, dict) and 'module' in config:
        assert config['name'] not in mm, f"name {config['name']} already exists."
        m = torch.nn.ModuleList(config['module'])
        config[config['name']] = m
        mm.update({config['name']: m})

    # optimizer
    elif isinstance(config, dict) and "params" in config and isinstance(config['params'], str):
        locals().update(**mm)
        config['params'] = eval(config['params'])


build(solver_dict, {}) 

# print(solver_dict)

for k in solver_dict:
    print(k, solver_dict[k])

# print(solver_dict['model']['module'][0])
# print(solver_dict)

print(solver_dict['reader'][0]['dataloader'].dataset.transforms)