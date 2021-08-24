from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype


import unittest
import numpy as np
import paddle
import paddle.nn as nn


def roi_pool(input,
             boxes,
             output_size,
             spatial_scale=1.0,
             boxes_num=None,
             name=None):
    """
    This operator implements the roi_pooling layer.
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).
    The operator has three steps:
        1. Dividing each region proposal into equal-sized sections with output_size(h, w);
        2. Finding the largest value in each section;
        3. Copying these max values to the output buffer.
    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn
    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        boxes (Tensor): boxes (Regions of Interest) to pool over. 
            2D-Tensor or 2D-LoDTensor with the shape of [num_boxes,4], the lod level is 1. 
            Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates, 
            and (x2, y2) is the bottom right coordinates.
        output_size (int or tuple[int, int]): The pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0
        boxes_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.
    Returns:
        Tensor: The pooled feature, 4D-Tensor with the shape of [num_boxes, C, output_size[0], output_size[1]].
    
    Examples:
        ..  code-block:: python
            import paddle
            x = paddle.rand([1, 256, 32, 32])
            boxes = paddle.rand([3, 4])
            boxes[:, 2] += boxes[:, 0] + 3
            boxes[:, 3] += boxes[:, 1] + 4
            boxes_num = paddle.to_tensor([3]).astype('int32')
            pool_out, argmaxes = roi_pool(x, boxes, boxes_num=boxes_num, output_size=3)
            assert pool_out.shape == [3, ] + x.shape[1:2] + [3, 3], ''
    """
    check_type(output_size, 'output_size', (int, tuple), 'roi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert boxes_num is not None, "boxes_num should not be None in dygraph mode."
        pool_out, argmaxes = core.ops.roi_pool(
            input, boxes, boxes_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale)
        return pool_out

    else:
        check_variable_and_dtype(input, 'input', ['float32'], 'roi_pool')
        check_variable_and_dtype(boxes, 'boxes', ['float32'], 'roi_pool')
        helper = LayerHelper('roi_pool', **locals())
        dtype = helper.input_dtype()
        pool_out = helper.create_variable_for_type_inference(dtype)
        argmaxes = helper.create_variable_for_type_inference(dtype='int32')

        inputs = {
            "X": input,
            "ROIs": boxes,
        }
        if boxes_num is not None:
            inputs['RoisNum'] = boxes_num
        helper.append_op(
            type="roi_pool",
            inputs=inputs,
            outputs={"Out": pool_out,
                     "Argmax": argmaxes},
            attrs={
                "pooled_height": pooled_height,
                "pooled_width": pooled_width,
                "spatial_scale": spatial_scale
            })
        return pool_out

    


class RoIPool(nn.Layer):
    def __init__(self, output_size, spatial_scale=1.):
        super(RoIPool, self).__init__()
        self._output_size = output_size
        self._spatial_scale = spatial_scale

    def forward(self, input, boxes, boxes_num):
        return roi_pool(input=input, boxes=boxes, output_size=self._output_size, spatial_scale=self._spatial_scale, boxes_num=boxes_num)
    
    def extra_repr(self):
        main_str = 'output_size={_output_size}, spatial_scale={_spatial_scale}'
        return main_str.format(**self.__dict__)
        

class TestRoIPool(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(1, 256, 32, 32).astype('float32')
        boxes = np.random.rand(3, 4)
        boxes[:, 2] += boxes[:, 0] + 3
        boxes[:, 3] += boxes[:, 1] + 4
        self.boxes = boxes.astype('float32')
        self.boxes_num = np.array([3], dtype=np.int32)

    def roi_pool_functional(self, output_size):

        if isinstance(output_size, int):
            output_shape = (3, 256, output_size, output_size)
        else:
            output_shape = (3, 256, output_size[0], output_size[1])

        if paddle.in_dynamic_mode():
            data = paddle.to_tensor(self.data)
            boxes = paddle.to_tensor(self.boxes)
            boxes_num = paddle.to_tensor(self.boxes_num)

            pool_out = roi_pool(
                data, boxes, boxes_num=boxes_num, output_size=output_size)
            np.testing.assert_equal(pool_out.shape, output_shape)

        else:
            data = paddle.static.data(
                shape=self.data.shape, dtype=self.data.dtype, name='data')
            boxes = paddle.static.data(
                shape=self.boxes.shape, dtype=self.boxes.dtype, name='boxes')
            boxes_num = paddle.static.data(
                shape=self.boxes_num.shape,
                dtype=self.boxes_num.dtype,
                name='boxes_num')

            pool_out = roi_pool(
                data, boxes, boxes_num=boxes_num, output_size=output_size)

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)

            pool_out = exe.run(paddle.static.default_main_program(),
                               feed={
                                   'data': self.data,
                                   'boxes': self.boxes,
                                   'boxes_num': self.boxes_num
                               },
                               fetch_list=[pool_out])

            np.testing.assert_equal(pool_out[0].shape, output_shape)

    def test_roi_pool_functional_dynamic(self):
        self.roi_pool_functional(3)
        self.roi_pool_functional(output_size=(3, 4))

    def test_roi_pool_functional_static(self):
        paddle.enable_static()
        self.roi_pool_functional(3)
        paddle.disable_static()

    def test_RoIPool(self):
        roi_pool_c = RoIPool(output_size=(4, 3))
        data = paddle.to_tensor(self.data)
        boxes = paddle.to_tensor(self.boxes)
        boxes_num = paddle.to_tensor(self.boxes_num)

        pool_out = roi_pool_c(data, boxes, boxes_num)
        np.testing.assert_equal(pool_out.shape, (3, 256, 4, 3))


    def test_value(self, ):
        data = np.array([i for i in range(1,17)]).reshape(1,1,4,4).astype(np.float32)
        boxes = np.array([[1., 1., 2., 2.], [1.5, 1.5, 3., 3.]]).astype(np.float32)
        boxes_num = np.array([2]).astype('int32')
        output = np.array([[[[11.]]], [[[16.]]]], dtype=np.float32)

        data = paddle.to_tensor(data)
        boxes = paddle.to_tensor(boxes)
        boxes_num = paddle.to_tensor(boxes_num)

        roi_pool_c = RoIPool(output_size=1)
        pool_out = roi_pool_c(data, boxes, boxes_num)
        np.testing.assert_almost_equal(pool_out.numpy(), output)


        
        
# class TestRoIPool_v1(unittest.TestCase):
#     def setUp(self):
#         self.data = np.random.rand(1, 256, 32, 32).astype('float32')
#         boxes = np.random.rand(3, 4)
#         boxes[:, 2] += boxes[:, 0] + 3
#         boxes[:, 3] += boxes[:, 1] + 4
#         self.boxes = boxes.astype('float32')
#         self.boxes_num = np.array([3], dtype=np.int32)
#         self.output_size = 3
#         self.output_value = None

#     @setattr
#     def output_shape(self, ):
#         if isinstance(self.output_size, int):
#             output_shape = (self.boxes_num, self.data.shape[1], self.output_size, self.output_size)
#         else:
#             output_shape = (self.boxes_num, self.data.shape[1], self.output_size[0], self.output_size[1])

#         return output_shape

#     def roi_pool_functional(self, ):

#         if paddle.in_dynamic_mode():
#             data = paddle.to_tensor(self.data)
#             boxes = paddle.to_tensor(self.boxes)
#             boxes_num = paddle.to_tensor(self.boxes_num)
#             output_size = self.output_size

#             pool_out = roi_pool(
#                 data, boxes, boxes_num=boxes_num, output_size=output_size)
#             np.testing.assert_equal(pool_out.shape, self.output_shape)
#             if self.output is not None:
#                 np.testing.assert_almost_equal(pool_out.numpy(), self.output_value, decimal=0.5)

#         else:
#             data = paddle.static.data(
#                 shape=self.data.shape, dtype=self.data.dtype, name='data')
#             boxes = paddle.static.data(
#                 shape=self.boxes.shape, dtype=self.boxes.dtype, name='boxes')
#             boxes_num = paddle.static.data(
#                 shape=self.boxes_num.shape,
#                 dtype=self.boxes_num.dtype,
#                 name='boxes_num')
#             output_size = self.output_size

#             pool_out = roi_pool(
#                 data, boxes, boxes_num=boxes_num, output_size=output_size)

#             place = paddle.CPUPlace()
#             exe = paddle.static.Executor(place)

#             pool_out = exe.run(paddle.static.default_main_program(),
#                                feed={
#                                    'data': self.data,
#                                    'boxes': self.boxes,
#                                    'boxes_num': self.boxes_num
#                                },
#                                fetch_list=[pool_out])

#             np.testing.assert_equal(pool_out[0].shape, self.output_shape)
#             if self.output is not None:
#                 np.testing.assert_almost_equal(pool_out[0], self.output_value, decimal=0.5)

#     def test_roi_pool_functional_dynamic(self):
#         self.output_size = (3, 4)
#         self.roi_pool_functional()
#         self.output_size = 3
#         self.roi_pool_functional()

#     def test_roi_pool_functional_static(self):
#         self.output_size = 3
#         paddle.enable_static()
#         self.roi_pool_functional()
#         paddle.disable_static()

#     def test_RoIPool(self, ):
#         self.output_size = (4, 3)
#         roi_pool_c = RoIPool(output_size=self.output_size)
#         data = paddle.to_tensor(self.data)
#         boxes = paddle.to_tensor(self.boxes)
#         boxes_num = paddle.to_tensor(self.boxes_num)

#         pool_out = roi_pool_c(data, boxes, boxes_num)
#         np.testing.assert_equal(pool_out.shape, self.output_shape)


#     def test_value(self, ):

#         self.data = np.array([i for i in range(1,17)]).reshape(1,1,4,4).astype('float32')
#         self.boxes = np.array([[1., 1., 2., 2.], [1.5, 1.5, 3., 3.]]).astype('float32')
#         self.boxes_num = np.array([2]).astype('int32')
#         self.output_size = 1
#         self.output = np.array([[[[11.]]], [[[16.]]]], dtype=float32)

#         self.roi_pool_functional()

#         paddle.enable_static()
#         self.roi_pool_functional()
#         paddle.disable_static()