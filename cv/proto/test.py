import torch

import cv_pb2 as cvpb
from google.protobuf import text_format
from google.protobuf import json_format
# from google.protobuf import pyext

import inspect
from collections import OrderedDict


solver = cvpb.Solver()
text_format.Merge(open('./sovler.prototxt', 'rb').read(), solver)

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
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __repr__(self) -> str:
        return f'Test {self.transforms}'

modules = {
    'Test': Test,
    'ToTensor': ToTensor,  
    'Resize': Resize,  
    'Conv2d': torch.nn.Conv2d,
    'ReLU': torch.nn.ReLU,
}

solver_dict = json_format.MessageToDict(solver, preserving_proto_field_name=True, including_default_value_fields=False)
# print(solver_dict)


def build(solver):
    '''build
    '''
    if not isinstance(solver, (dict, list)):
        return

    for i, k in enumerate(solver):
        v = solver[k] if isinstance(solver, dict) else k
        m = build(v)

        if m is not None:
            m.top = v.get('top', None)
            m.bottom = v.get('bottom', None)

            solver[(k if isinstance(solver, dict) else i)] = m

    if isinstance(solver, dict) and 'type' in solver:
        m = build_module(modules[solver['type']], solver)
        return m

build(solver_dict) 

# for k in solver_dict:
#     print(k, solver_dict[k])

print(solver_dict['model']['module'][0])