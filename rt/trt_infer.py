
import os
import time
import glob
import contextlib
from collections import namedtuple, OrderedDict

import numpy as np
from PIL import Image 

import tensorrt as trt

import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T 
import torchvision.transforms.functional as F 

from typing import List, Tuple

# from utils import *

class TRTInference(object):
    def __init__(self, engine_path='dino.engine', device='cuda:0', backend='torch', onnx_path='', max_batch_size=32, verbose=False):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        
        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)  

        self.engine = self.load_engine(engine_path)

        # if engine_path:
        #     self.engine = self.load_engine(engine_path)
        # else:
        #     self.engine = self.build_engine(onnx_path, engine_path)

        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.context, self.max_batch_size, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def init(self, ):
        self.dynamic = False 

        
    def load_engine(self, path):
        '''load engine
        '''
        trt.init_libnvinfer_plugins(self.logger, '')
        with open(path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    
    def get_input_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    

    def get_output_names(self, ):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    
    def get_bindings(self, engine, context, max_batch_size=32, device=None):
        '''build binddings
        '''
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        # max_batch_size = 1

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                dynamic = True 
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
                    context.set_input_shape(name, shape)

            # if -1 in shape:
            #     dynamic = True 
            #     if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:  # dynamic
            #         shape = engine.get_tensor_profile_shape(name, i)[2]
            #         context.set_input_shape(name, shape)
            #         max_batch_size = shape[0]
            #     else:
            #         shape[0] = max_batch_size

            if self.backend == 'cuda':
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    data = np.random.randn(*shape).astype(dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 
                else:
                    data = cuda.pagelocked_empty(trt.volume(shape), dtype)
                    ptr = cuda.mem_alloc(data.nbytes)
                    bindings[name] = Binding(name, dtype, shape, data, ptr) 

            else:
                data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    
    def run_torch(self, blob):
        '''torch input
        '''
        for n in self.input_names:
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape) 
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs


    def async_run_cuda(self, blob):
        '''numpy input
        '''
        for n in self.input_names:
            cuda.memcpy_htod_async(self.bindings_addr[n], blob[n], self.stream)
        
        bindings_addr = [int(v) for _, v in self.bindings_addr.items()]
        self.context.execute_async_v2(bindings=bindings_addr, stream_handle=self.stream.handle)
        
        outputs = {}
        for n in self.output_names:
            cuda.memcpy_dtoh_async(self.bindings[n].data, self.bindings[n].ptr, self.stream)
            outputs[n] = self.bindings[n].data
        
        self.stream.synchronize()
        
        return outputs
    

    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)
                
    
    def synchronize(self, ):
        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.synchronize()

        elif self.backend == 'cuda':
            self.stream.synchronize()
    

    def warmup(self, blob, n):
        for _ in range(n):
            _ = self(blob)


    def speed(self, blob, n):
        self.time_profile.reset()
        for _ in range(n):
            with self.time_profile:
                _ = self(blob)

        return self.time_profile.total / n 


    def build_engine(self, onnx_file_path, engine_file_path, max_batch_size=32):
        '''Takes an ONNX file and creates a TensorRT engine to run inference with
        http://gitlab.baidu.com/paddle-inference/benchmark/blob/main/backend_trt.py#L57
        '''
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        with trt.Builder(self.logger) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, self.logger) as parser, \
            builder.create_builder_config() as config:
            
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1024 MiB
            config.set_flag(trt.BuilderFlag.FP16)

            with open(onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # import onnx 
            # model = onnx.load(onnx_file_path)
            # input_all = [node.name for node in model.graph.input]
            # input_initializer = [node.name for node in model.graph.initializer]
            # input_names = list(set(input_all)  - set(input_initializer))
            # input_nodes = [node for node in model.graph.input if node.name in input_name]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            outputs = [network.get_output(i) for i in range(network.num_outputs)]
            
            # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/exporter.py#L482
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                shape = network.get_input(i).shape
                if shape[0] == -1:
                    profile.set_shape(network.get_input(i).name, (1, *shape[1:]), (max_batch_size//2, *shape[1:]), (max_batch_size, *shape[1:]))
            config.add_optimization_profile(profile)


            # # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html
            # for i in range(network.num_inputs):
            #     shape = network.get_input(i).shape
            #     if shape[0] == -1:
            #         network.get_input(i).shape = tuple([max_batch_size, ] + list(shape)[1:])

            serialized_engine = builder.build_serialized_network(network, config)
            with open(engine_file_path, 'wb') as f:
                f.write(serialized_engine)

            return serialized_engine


# ----------


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self, ):
        self.total = 0
        
    def __enter__(self, ):
        self.start = self.time()
        return self 
    
    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start
    
    def reset(self, ):
        self.total = 0
    
    def time(self, ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()



class ToTensor(T.ToTensor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic 
        return super().__call__(pic)


class PadToSize(T.Pad):
    def __init__(self, size, fill=0, padding_mode='constant'):
        super().__init__(0, fill, padding_mode)
        self.size = size
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.
        """
        w, h = F.get_image_size(img)
        padding = (0, 0, self.size[0] - w, self.size[1] - h)
        return F.pad(img, padding, self.fill, self.padding_mode)


class Dataset(data.Dataset):
    def __init__(self, img_dir: str='', preprocess: T.Compose=None, device='cuda:0') -> None:
        super().__init__()

        self.device = device
        self.im_path_list = list(glob.glob(os.path.join(img_dir, '*.jpg')))

        if preprocess is None: 
            self.preprocess = T.Compose([
                    T.Resize(size=639, max_size=640),
                    PadToSize(size=(640, 640), fill=114),
                    ToTensor(),
                    T.ConvertImageDtype(torch.float),
            ])
        else:
            self.preprocess = preprocess

    def __len__(self, ):
        return len(self.im_path_list)


    def __getitem__(self, index):
        # im = Image.open(self.img_path_list[index]).convert('RGB')
        im = torchvision.io.read_file(self.im_path_list[index])
        im = torchvision.io.decode_jpeg(im, mode=torchvision.io.ImageReadMode.RGB, device=self.device)
        orig_shape = im.shape # c,h,w

        im = self.preprocess(im)
        cur_shape = im.shape 

        blob = {
            'image': im, 
            'im_shape': torch.tensor([640., 640.]).to(im.device),
            'scale_factor': torch.tensor([1., 1.]).to(im.device),
            'orig_size': torch.tensor([orig_shape[2], orig_shape[1]]).to(im.device),
            # 'ratio': torch.tensor().to(im.device),
        }

        return blob


    @staticmethod
    def post_process():
        pass

    @staticmethod
    def collate_fn():
        pass


def draw_result_ppdetr(m, blob):
    '''show result
    '''
    outputs = m(blob)
    preds = outputs.values()[0]
    preds = preds[preds[:, 1] > 0.5]

    im = (blob['image'][0] * 255).to(torch.uint8)
    im = torchvision.utils.draw_bounding_boxes(im, boxes=preds[:, 2:], width=2)
    # torchvision.utils.save_image(im, 'test.jpg')
    Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{i}.jpg')



def draw_result_yolo(blob, outputs, draw_score_threshold=0.25, name=''):
    '''show result
    Keys:
        'num_dets', 'det_boxes', 'det_scores', 'det_classes'
    '''    
    for i in range(blob['image'].shape[0]):
        det_scores = outputs['det_scores'][i]
        det_boxes = outputs['det_boxes'][i][det_scores > draw_score_threshold]
        
        im = (blob['image'][i] * 255).to(torch.uint8)
        im = torchvision.utils.draw_bounding_boxes(im, boxes=det_boxes, width=2)
        Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{name}_{i}.jpg')


def save_result_json(blob, outputs, path='result.json'):
    import json 
    results = []

    for i in range(blob['image'].shape[0]):
        det_scores = outputs['det_scores'][i]
        det_boxes = outputs['det_boxes'][i]

        for j in range(len(det_scores)):
            results.append({
                'image_id': None,
                'category_id': None,
                'bbox': det_boxes[j],
                'score': det_scores[j], 
            })

    with open(path, 'wb') as f:
        json.dump(results, f)


def dummy_blob(batch_size=1, backend='torch'):
    '''
    '''
    if backend == 'torch':
        blob = {
            'image': torch.rand(batch_size, 3, 640, 640).to('cuda:0'),
            'im_shape': torch.tensor([[640., 640.]]).tile(batch_size, 1).to('cuda:0'),
            'scale_factor': torch.tensor([[1., 1.]]).tile(batch_size, 1).to('cuda:0'),
        }
        
    else:
        blob = {
            'image': np.random.rand(batch_size, 3, 640, 640).astype(np.float32),
            'im_shape': np.array([[640., 640.]]).repeat(batch_size, 0).astype(np.float32),
            'scale_factor': np.array([[1., 1.]]).repeat(batch_size, 0).astype(np.float32),
        }
    return blob


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--backend', type=str, default='torch')
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--img_dir', type=str, default='')
    # parser.add_argument('--img_file', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--max_batch_size', type=int, default=32)

    args = parser.parse_args()

    m = TRTInference(args.path, backend=args.backend, device=f'cuda:{args.device_id}', max_batch_size=args.max_batch_size)
    
    preprocess = T.Compose([
        T.Resize(size=(640, 640)),
        ToTensor(),
        T.ConvertImageDtype(torch.float),
    ])

    dataset = Dataset(img_dir=args.img_dir, preprocess=preprocess)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    blob = dummy_blob(args.batch_size, backend=args.backend)
    m.warmup(blob, args.warmup_steps)

    time_profile = TimeProfiler()
    # times = []

    for i, blob in enumerate(dataloader):
        # t = m.speed(blob, n=args.repeats)
        # times.append(t)

        with time_profile:
            _ = m.warmup(blob, n=args.repeats)

        if args.draw:
            draw_result_yolo(blob, m(blob), 0.25, i)

    # print(np.mean(times) * 1000)
    print(f'total time {time_profile.total} for {len(dataset)} images with batch_size={args.batch_size}', )

    print('fps: ', 1. / (time_profile.total / args.repeats / len(dataset)))

    print(time_profile.total / len(dataloader) / args.repeats * 1000)    

