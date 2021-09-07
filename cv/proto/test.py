import torch
import torchvision

from google.protobuf import text_format
from google.protobuf import json_format
from google.protobuf import reflection
# from google.protobuf import pyext

import copy
import inspect
import functools
from types import SimpleNamespace
from collections import OrderedDict
from typing import Optional, Iterable, Sequence, Union
assert Union[str, None] == Optional[str], ''

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


class CocoEval(object):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold

    def __call__(self, ):
        pass

    def format_ouput(self, ):
        raise NotImplementedError('')


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



class YOLO(torch.nn.Module):
    def __init__(self, backbone, neck, head, postprocess):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head 
        self.postprocess = postprocess
    
    def forward(self, x):
        pass


class Conv2d(torch.nn.Conv2d):
    def __init__(self,      
                in_channels: int,
                out_channels: int,
                kernel_size,
                stride = 1,
                padding = 0,
                dilation = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',  # TODO: refine this type
                device=None,
                dtype=None) -> None:

        if isinstance(kernel_size, Sequence) and len(kernel_size) == 1:
            kernel_size = kernel_size[0]
        if isinstance(stride, Sequence) and len(stride) == 1:
            stride = stride[0]
        if isinstance(padding, Sequence) and len(padding) == 1:
            padding = padding[0]

        super().__init__(in_channels, 
                         out_channels, 
                         kernel_size, 
                         stride=stride, 
                         padding=padding, 
                         dilation=dilation, 
                         groups=groups, 
                         bias=bias, 
                         padding_mode=padding_mode, 
                         device=device, 
                         dtype=dtype)



modules.update({
    'Test': Test,
    'ToTensor': ToTensor,  
    'Resize': Resize,  
    'Conv2d': torch.nn.Conv2d,
    'ReLU': torch.nn.ReLU,
    'CocoDet': CocoDet,
    'CocoEval': CocoEval,
    'DataLoader': torch.utils.data.DataLoader,
    'Compose': Compose,
    'Mosaic': Mosaic,
    'ModuleList': ModuleList,
    'YOLO': YOLO,
    'Conv2d': Conv2d,

})



def dict_deep_merge(*dicts, add_new_key=True):
    '''merge
    https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    '''
    if len(dicts) == 1:
        assert isinstance(dicts[0], dict), ''
        return dicts[0]

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

            if 'top' in v or 'bottom' in v:
                assert not (hasattr(m, 'top') or hasattr(m, 'bottom')), f'{m} .top, .bottom'
                m.top = v.get('top', None)
                m.bottom = v.get('bottom', None)

            if 'pretrained' in v and isinstance(m, torch.nn.Module):
                # TODO
                # m.load_state_dict(torch.load(v))
                pass

            if 'name' in v:
                assert v['name'] not in mm, f"name {v['name']} already exists."
                mm.update({v['name']: m})

            config[k if isinstance(config, dict) else i] = m

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

    



class SolverProto(object):
    def __init__(self, path) -> None:
        self.path = path
        self.merged_dict = self.parse(path)
        self.merged_proto = self.ParseDict(self.merged_dict) 
        # cvpb.Solver()
        # json_format.ParseDict(self.merged_dict, self.merged_proto)

    def parse(self, path):
        '''parse
        '''
        solver = self.ParseFile(path)
        solver_dict = self.MessageToDict(solver)

        solver_dicts = [self.parse(path) for path in solver.include]
        solver_dicts += [solver_dict, ]

        # for path in solver.include:
        #     solver_dicts.insert(0, self.parse(path))

        merged_dict = dict_deep_merge(*solver_dicts, add_new_key=True)

        merged_proto = self.ParseDict(merged_dict)

        merged_dict = self.MessageToDict(merged_proto)
        
        return merged_dict


    def build(self, ):
        '''build
        '''
        config = copy.deepcopy(self.merged_dict)
        mm = {}
        build(config, mm)

        config = SimpleNamespace(**config)
        config.modules = mm

        return config
    
    @staticmethod
    def MessageToDict(message):
        return json_format.MessageToDict(message, 
                                         preserving_proto_field_name=True, 
                                         including_default_value_fields=False, 
                                         use_integers_for_enums=True)

    @staticmethod
    def ParseDict(message_dict):
        message = cvpb.Solver()
        json_format.ParseDict(message_dict, message)
        return message

    @staticmethod
    def ParseFile(path):
        message = cvpb.Solver()
        text_format.Parse(open(path, 'rb').read(), message)
        return message

    
solver = cvpb.Solver()
text_format.Merge(open('./sovler.prototxt', 'rb').read(), solver)
# print(solver)

# model = cvpb.Solver()
# text_format.Merge(open('./model.prototxt', 'rb').read(), model)
# print(model)

# configs = [solver, ]
configs = []

for path in solver.include:
    config = cvpb.Solver()
    text_format.Merge(open(path, 'rb').read(), config)
    # TODO
    configs.append(config)

configs.append(solver)

configs_dict = [json_format.MessageToDict(config, \
    preserving_proto_field_name=True, 
    including_default_value_fields=False,  
    use_integers_for_enums=True) for config in configs]
    
# print(configs_dict)
print('------')






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

# keep order in proto.
s = cvpb.Solver()
json_format.ParseDict(r, s)
r = json_format.MessageToDict(s, \
    preserving_proto_field_name=True, 
    including_default_value_fields=False, 
    use_integers_for_enums=True)


mm = {}
build(r, mm) 
print(r)
print('----')

r = SimpleNamespace(**r)
print(r.reader)
print(r.model)
print(r.optimizer)
print(r.runtime)


# for k in solver_dict:
#     print(k, solver_dict[k])

# print(solver_dict['model']['module'][0])
# print(solver_dict['reader'][0]['dataloader'].dataset.transforms)

# print('-----')


# var = SimpleNamespace(**mm)
# print(var.yolo)
# print(var.dataloader1)
# print(var.coco_eval)

solver = SolverProto('./sovler.prototxt')
# # print(solver.prototxt)
# print(solver.merged_dict)

solver = solver.build()
print(list(solver.modules.keys()))
