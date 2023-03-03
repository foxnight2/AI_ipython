# https://github.com/lyuwenyu/PaddleDetection/blob/vit_backbone_speed_L/configs/vitdet/vit.py


# from ppdet.core.workspace import load_config, merge_config
# from ppdet.core.workspace import create

import paddle

# cfg = load_config('./configs/vitdet/vit.yml')
# model = create('VisionTransformer')

# data = paddle.rand([1, 3, 640, 640])
# model(data)
# print(model)

from paddle.inference import Config
from paddle.inference import create_predictor
from paddle.inference import PrecisionType
import numpy as np

image = paddle.static.InputSpec(
    shape=[1, 3, 640, 640], dtype='float32', name="image")
# paddle.onnx.export(model, 'vit.onnx', input_spec=[{'image': image}], opset_version=12, )

static_model = paddle.jit.to_static(model, input_spec=[{'image': image}])
paddle.jit.save(
    static_model,
    'test/model', )

config = Config('test/model.pdmodel', 'test/model.pdiparams')
config.enable_use_gpu(500, 0)
config.switch_ir_optim()
config.enable_memory_optim()
# config.collect_shape_range_info(tuned_trt_shape_file)

config.enable_tensorrt_engine(
    workspace_size=1 << 30,
    precision_mode=PrecisionType.Half,
    # precision_mode=PrecisionType.Float32,
    max_batch_size=1,
    min_subgraph_size=5,
    use_static=False,
    use_calib_mode=False)

config.collect_shape_range_info('shape_range_info.pbtxt')  # only once, xx

config.enable_tuned_tensorrt_dynamic_shape('shape_range_info.pbtxt',
                                           True)  # keep 

predictor = create_predictor(config)

img = np.random.rand(
    1,
    3,
    640,
    640, ).astype(np.float32)

input_names = predictor.get_input_names()
input_tensor = predictor.get_input_handle(input_names[0])
input_tensor.reshape(img.shape)
input_tensor.copy_from_cpu(img.copy())

predictor.run()

import time
tic = time.time()
predictor.run()
toc = time.time() - tic

output_names = predictor.get_output_names()
output_tensor = predictor.get_output_handle(output_names[0])
output_data = output_tensor.copy_to_cpu()