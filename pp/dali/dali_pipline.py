


from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def

import nvidia.dali.fn as fn
import nvidia.dali.types as types


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
