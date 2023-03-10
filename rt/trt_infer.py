
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



class TRTInference(object):
    def __init__(self, path='dino.engine', device='cuda:0', backend='torch'):
        self.path = path
        self.device = device
        self.backend = backend
        
        self.engine = self.load_engine(path)
        self.context = self.engine.create_execution_context()

        self.bindings = self.get_bindings(self.engine, self.device)
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        
        if self.backend == 'cuda':
            self.stream = cuda.Stream()

        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        '''load engine
        '''
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, '')

        with open(path, 'rb') as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    
    
    def get_input_names(self, ):
        names = []
        for i, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names
    

    def get_output_names(self, ):
        names = []
        for i, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    
    def get_bindings(self, engine, device=None):
        '''build binddings
        '''
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        bindings = OrderedDict()
        
        for _, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if -1 in shape:  # dynamic
                dynamic = True

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


    def __call__(self, blob):
        if self.backend == 'torch':
            return self.run_torch(blob)

        elif self.backend == 'cuda':
            return self.async_run_cuda(blob)
                

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
    def __init__(self, size, fill=0, padding_mode="constant"):
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

        im = self.preprocess(im)
        
        blob = {
            'image': im, 
            'im_shape': torch.tensor([640., 640.]).to(im.device),
            'scale_factor': torch.tensor([1., 1.]).to(im.device),
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
    preds = outputs['reshape2_94.tmp_0']
    preds = preds[preds[:, 1] > 0.5]

    im = (blob['image'][0] * 255).to(torch.uint8)
    im = torchvision.utils.draw_bounding_boxes(im, boxes=preds[:, 2:], width=2)
    # torchvision.utils.save_image(im, 'test.jpg')
    Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{i}.jpg')



def draw_result_yolo(outputs, name=''):
    '''show result
    Keys:
        'num_dets', 'det_boxes', 'det_scores', 'det_classes'
    '''
    # outputs = m(blob)
    # print(outputs['num_dets'].shape)
    
    for i in range(outputs['num_dets'].shape[0]):
        det_scores = outputs['det_scores'][i]
        det_boxes = outputs['det_boxes'][i][det_scores > 0.25]
        # print(det_boxes)
        
        im = (blob['image'][i] * 255).to(torch.uint8)
        im = torchvision.utils.draw_bounding_boxes(im, boxes=det_boxes, width=2)
        Image.fromarray(im.permute(1, 2, 0).cpu().numpy()).save(f'test_{name}_{i}.jpg')



def dummy_blob(backend='torch'):
    '''
    '''
    if backend == 'torch':
        blob = {
            'image': torch.rand(1, 3, 640, 640).to('cuda:0'),
            'im_shape': torch.tensor([[640., 640.]]).to('cuda:0'),
            'scale_factor': torch.tensor([[1., 1.]]).to('cuda:0'),
        }
        
    else:
        blob = {
            'image': np.random.rand(1, 3, 640, 640).astype(np.float32),
            'im_shape': np.array([[640., 640.]]).astype(np.float32),
            'scale_factor': np.array([[1., 1.]]).astype(np.float32),
        }
    return blob


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--backend', type=str, default='torch')
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--img_dir', type=str, default='')
    args = parser.parse_args()



    m = TRTInference(args.path, backend=args.backend)
    
    preprocess = T.Compose([
        T.Resize(size=(640, 640)),
        ToTensor(),
        T.ConvertImageDtype(torch.float),
    ])

    dataset = Dataset(img_dir=args.img_dir, preprocess=preprocess)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)


    blob = dummy_blob(backend=args.backend)
    m.warmup(blob, args.warmup_steps)

    time_profile = TimeProfiler()
    # times = []

    for i, blob in enumerate(dataloader):
        # t = m.speed(blob, n=args.repeats)
        # times.append(t)

        with time_profile:
            _ = m.warmup(blob, n=args.repeats)


        outputs = m(blob)
        draw_result_yolo(outputs, i)

    # print(np.mean(times) * 1000)
    print(time_profile.total / len(dataloader) / args.repeats * 1000)    

