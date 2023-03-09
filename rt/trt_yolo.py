import torch 

import numpy as np 
import onnxruntime as ort 

from collections import OrderedDict


# ../software/TensorRT-8.5.1.7/bin/trtexec --onnx=./yolov8n_w_nms.onnx --saveEngine=yolov8n_w_nms.engine --buildOnly --fp16 
# http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnxops/onnx__EfficientNMS_TRT.html

class YOLOL(torch.nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        from ultralytics import YOLO
        # Load a model
        # build a new model from scratch
        # model = YOLO(f'{name}.yaml')  

        # load a pretrained model (recommended for training)
        model = YOLO(f'{name}.pt')  
        self.model = model.model

    def forward(self, x):
        pred: torch.Tensor = self.model(x)[0] # n 84 8400,
        pred = pred.permute(0, 2, 1)
        boxes, scores = pred.split([4, 80], dim=-1)
        print(boxes.shape, scores.shape)
        return boxes, scores



def export_yolov8_onnx(name='yolov8n'):
    m = YOLOL(name)

    x = torch.rand(1, 3, 640, 640)

    dynamic_axes = {
        # 'image': {0: 'N', 2: 'H', 3: 'W'}
        'image': {0: 'N'}
    }

    torch.onnx.export(m, x, f'{name}.onnx', input_names=['image'], output_names=['boxes', 'scores'], opset_version=13, dynamic_axes=dynamic_axes)
    # torch.onnx.export(m, x, f'{name}.onnx', input_names=['images'], opset_version=13)
    # output_names=['boxes', 'scores'], 

def onnx_insert_nms(name):
    import onnx
    import onnx_graphsurgeon
    from onnxsim import simplify

    onnx_model = onnx.load(f'{name}.onnx')
    onnx_model, _ = simplify(onnx_model,  overwrite_input_shapes={'image': [1, 3, 640, 640]})

    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = 300
    attrs = OrderedDict(
        plugin_version='1',
        background_class=-1,
        max_output_boxes=topk,
        score_threshold=0.001,
        iou_threshold=0.7,
        score_activation=False,
        box_coding=0, )

    outputs = [
        onnx_graphsurgeon.Variable('num_dets', np.int32, [-1, 1]),
        onnx_graphsurgeon.Variable('det_boxes', np.float32, [-1, topk, 4]),
        onnx_graphsurgeon.Variable('det_scores', np.float32, [-1, topk]),
        onnx_graphsurgeon.Variable('det_classes', np.int32, [-1, topk])
    ]

    for out in graph.outputs:
        print(out)

    graph.layer(
        op='EfficientNMS_TRT',
        name="batched_nms",
        inputs=[graph.outputs[0], graph.outputs[1]],
        outputs=outputs,
        attrs=attrs)

    graph.outputs = outputs
    graph.cleanup().toposort()

    onnx.save(onnx_graphsurgeon.export_onnx(graph), f'{name}_w_nms.onnx')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='yolov8l')
    args = parser.parse_args()

    export_yolov8_onnx(name=args.name)
    onnx_insert_nms(name=args.name)

    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    sess = ort.InferenceSession(f'{args.name}_w_nms.onnx')
    output = sess.run(output_names=None, input_feed={'image': data})
    