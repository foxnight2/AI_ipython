import torch

import cv_pb2 as cvpb
from google.protobuf import text_format
from google.protobuf import json_format
# from google.protobuf import pyext

import inspect
from collections import OrderedDict


solver = cvpb.Solver()
text_format.Merge(open('./sovler.prototxt', 'rb').read(), solver)
print(solver)

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

class Test():
    def __init__(self, m) -> None:
        self.m = m

    def __repr__(self) -> str:
        return f'Test {(self.m)}'


modules = {m: getattr(torch.nn, m) for m in dir(torch.nn) if inspect.isclass(getattr(torch.nn, m)) and issubclass(getattr(torch.nn, m), torch.nn.Module)}

modules.update({
    'Test': Test,
    'ToTensor': ToTensor,  
    'Resize': Resize,  
    'Conv2d': torch.nn.Conv2d,
    'ReLU': torch.nn.ReLU,
})


solver_dict = json_format.MessageToDict(solver, preserving_proto_field_name=True, including_default_value_fields=False)
# print(solver_dict)


def hasattr_and_not_none(obj, name):
    '''hasattr_and_not_none
    '''
    return hasattr(obj, name) and getattr(obj, name) is not None


def merge():
    '''merge
    '''
    pass


def build(solver, mm):
    '''build
    '''
    if not isinstance(solver, (dict, list)):
        return

    for i, k in enumerate(solver):
        v = solver[k] if isinstance(solver, dict) else k
        m = build(v, mm)

        if m is not None:
            assert not (hasattr(m, 'top') or hasattr(m, 'bottom')), ''
            m.top = v.get('top', None)
            m.bottom = v.get('bottom', None)
            solver[(k if isinstance(solver, dict) else i)] = m

            mm.update({v['name']: m} if 'name' in v else {})

    if isinstance(solver, dict) and 'type' in solver:
        k = [k for k in solver if 'param' in k]
        v = solver[k[0]] if len(k) != 0 else {}
        # TODO
        v.update({_k: mm[_v] for _k, _v in v.items() if isinstance(_v, str) and _v in mm})
        m = build_module(modules[solver['type']], v)

        return m

# mm = {}
build(solver_dict, {}) 

print(solver_dict)

# for k in solver_dict:
#     print(k, solver_dict[k])

# print(solver_dict['model']['module'][0])
# print(solver_dict)
