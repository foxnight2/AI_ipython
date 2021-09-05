import torch
import torchvision

from google.protobuf import text_format
from google.protobuf import json_format
from google.protobuf import reflection
# from google.protobuf import pyext

import copy
import inspect
from types import SimpleNamespace
from collections import OrderedDict
from typing import Optional, Iterable

import cv_pb2 as cvpb


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




modules = {m: getattr(torch.nn, m) for m in dir(torch.nn) \
    if inspect.isclass(getattr(torch.nn, m)) and issubclass(getattr(torch.nn, m), torch.nn.Module)}

modules.update({m: getattr(torch.optim, m) for m in dir(torch.optim) \
    if inspect.isclass(getattr(torch.optim, m)) and issubclass(getattr(torch.optim, m), torch.optim.Optimizer)})

modules.update({m: getattr(torch.optim.lr_scheduler, m) for m in dir(torch.optim.lr_scheduler) \
    if inspect.isclass(getattr(torch.optim.lr_scheduler, m)) })


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

class ModuleList(torch.nn.ModuleList):
    def __init__(self, module: Optional[Iterable[torch.nn.Module]]) -> None:
        super().__init__(modules=module)


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
    'ModuleList': ModuleList,
})



def dict_deep_merge(*dicts, add_new_key=True):
    '''merge
    https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    '''
    r = copy.deepcopy(dicts[0]) 

    for d in dicts[1: ]:

        assert isinstance(d, dict), ''
        if add_new_key is False:
            d = {k: d[k] for k in set(r).intersection(set(d)) }

        for k in d: 
            if r.get(k) is None:
                r[k] = d[k]

            elif k in r and type(r[k]) != type(d[k]):
                raise RuntimeError('')

            elif isinstance(r[k], dict) and isinstance(d[k], dict):
                r[k] = dict_deep_merge(r[k], d[k], add_new_key=add_new_key)

            elif isinstance(r[k], list) and isinstance(d[k], list):
                if not isinstance(d[k][0], dict):
                    r[k] = d[k] 
                else:
                    # TODO
                    # n = min(len(r[k]), len(d[k]))
                    # r[k][:n] = [dict_deep_merge(_r, _d, add_new_key=add_new_key) for _r, _d in zip(r[k], d[k])]
                    # r[k].extend(d[k][n:])
                    # assert items in `list` must have `name` field.
                    names = [x['name'] for x in r[k]]
                    for x in d[k]:
                        if x['name'] in names:
                            i = names.index(x['name'])
                            r[k][i] = dict_deep_merge(r[k][i], x, add_new_key=add_new_key)
                        else:
                            r[k].append(x)

            else:
                r[k] = d[k]

    return r


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

            if 'name' in v:
                assert v['name'] not in mm, f"name {v['name']} already exists."
                mm.update({v['name']: m})
    
    # module
    if isinstance(config, dict) and 'type' in config:
        k = [k for k in config if '_param' in k]
        v = config[k[0]] if len(k) != 0 else {}
        v.update({_k: mm[_v] for _k, _v in v.items() if isinstance(_v, str) and _v in mm})

        m = build_module(modules[config['type']], v)

        return m

    # optimizer
    elif isinstance(config, dict) and "params" in config and isinstance(config['params'], str):
        locals().update(**mm)
        config['params'] = eval(config['params'])






solver = cvpb.Solver()
text_format.Merge(open('./sovler.prototxt', 'rb').read(), solver)
print(solver)

# configs = [solver, ]
configs = []

for path in solver.include:
    config = cvpb.Solver()
    text_format.Merge(open(path, 'rb').read(), config)
    # TODO
    configs.append(config)

configs.append(solver)

configs_dict = [json_format.MessageToDict(config, preserving_proto_field_name=True, including_default_value_fields=False) for config in configs]

# optimizer = cvpb.Solver()
# text_format.Merge(open('./optimizer.prototxt', 'rb').read(), optimizer)
# reader = cvpb.Solver()
# text_format.Merge(open('./optimizer.prototxt', 'rb').read(), reader)

# print(optimizer)
# c += 1

# solver.MergeFrom(optimizer)
# print(solver)
# c+=1

# solver_dict = json_format.MessageToDict(solver, 
#                                         preserving_proto_field_name=True, 
#                                         including_default_value_fields=False)
# optim_dict = json_format.MessageToDict(optimizer, 
#                                         preserving_proto_field_name=True, 
#                                         including_default_value_fields=False)
# reader_dict = json_format.MessageToDict(reader, 
#                                         preserving_proto_field_name=True, 
#                                         including_default_value_fields=False)

# print(solver_dict)
# print(optim_dict)


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

# r = dict_deep_merge(solver_dict, optim_dict, reader_dict, add_new_key=True)
r = dict_deep_merge(*configs_dict, add_new_key=True)

# print(r)
# print()

build(r, {}) 
print(r)

# for k in solver_dict:
#     print(k, solver_dict[k])

# print(solver_dict['model']['module'][0])
# print(solver_dict['reader'][0]['dataloader'].dataset.transforms)

# print('-----')
var = SimpleNamespace(**r)
print(var.reader)