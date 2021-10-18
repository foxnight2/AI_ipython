

import numpy as np

from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from nvidia.dali.plugin.pytorch import DALIGenericIterator


image_dir = './images'
max_batch_size = 1


@pipeline_def
def simple_pipeline():
    files, labels = fn.readers.file(file_root=image_dir, random_shuffle=True)
    # images = fn.decoders.image(files, device='cpu')
    images = fn.decoders.image(files, device='mixed')
    
    angle = fn.random.uniform(range=(-10.0, 10.0))
    images = fn.rotate(images, angle=angle, fill_value=0)

    return images, labels.gpu()


pipe = simple_pipeline(batch_size=max_batch_size, num_threads=2, device_id=0, seed=1234)
pipe.build()

imgs, labs = pipe.run()

print(imgs, imgs.is_dense_tensor(), imgs.as_tensor().shape(), )
print(labs, labs.is_dense_tensor(), labs.as_tensor().shape(), )
print(labs.as_cpu())

print('-----------')



@pipeline_def
def torch_pipeline():
    device_id = Pipeline.current().device_id
    
    files, labels = fn.readers.file(file_root=image_dir, random_shuffle=True)
    
    images = fn.decoders.image(files, device='mixed')
    images = fn.resize(images, resize_shorter=fn.random.uniform(range=(256, 480)), interp_type=types.INTERP_LINEAR)
    
    return images, labels.gpu()


pipes = [torch_pipeline(batch_size=1, num_threads=2, device_id=device_id) for device_id in range(2)]
for p in pipes:
    p.build()
    
dali_iter = DALIGenericIterator(pipes, ['data', 'label'], reader_name='Reader')


for i, data in enumerate(dali_iter):

    print(type(data))
    
    for d in data:
        label = d["label"]
        image = d["data"]
        
        ## labels need to be integers
        assert(np.equal(np.mod(label, 1), 0).all())
        ## labels need to be in range pipe_name[2]
        assert((label >= label_range[0]).all())
        assert((label <= label_range[1]).all())

        
        
# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/detection_pipeline.html