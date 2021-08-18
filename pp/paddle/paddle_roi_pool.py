from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype


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
        return pool_out, argmaxes

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
        return pool_out, argmaxes
