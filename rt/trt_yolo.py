import torch 
import torchvision

import numpy as np 
import onnxruntime as ort 

from collections import OrderedDict

from PIL import Image



class YOLOv8(torch.nn.Module):
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
        '''https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216
        '''
        pred: torch.Tensor = self.model(x)[0] # n 84 8400,
        pred = pred.permute(0, 2, 1)
        boxes, scores = pred.split([4, 80], dim=-1)
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        return boxes, scores





def export_onnx(name='yolov8n'):
    '''export onnx
    '''
    m = YOLOv8(name)

    x = torch.rand(1, 3, 640, 640)
    dynamic_axes = {
        # 'image': {0: 'N', 2: 'H', 3: 'W'},
        # 'image': {0: 'N'},
        'image': {0: '-1'}
    }

    torch.onnx.export(m, x, f'{name}.onnx', input_names=['image'], output_names=['boxes', 'scores'], opset_version=13, dynamic_axes=dynamic_axes)
    # torch.onnx.export(m, x, f'{name}.onnx', input_names=['images'], opset_version=13)
    # output_names=['boxes', 'scores'], 

    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    sess = ort.InferenceSession(f'{name}.onnx')
    _ = sess.run(output_names=None, input_feed={'image': data})
    

def onnx_insert_nms(name, score_threshold=0.01, iou_threshold=0.7, max_output_boxes=300, simplify=False):
    '''http://www.xavierdupre.fr/app/onnxcustom/helpsphinx/api/onnxops/onnx__EfficientNMS_TRT.html
    '''
    import onnx
    import onnx_graphsurgeon
    from onnxsim import simplify

    onnx_model = onnx.load(f'{name}.onnx')
    if simplify:
        onnx_model, _ = simplify(onnx_model,  overwrite_input_shapes={'image': [1, 3, 640, 640]})

    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    topk = max_output_boxes
    attrs = OrderedDict(
        plugin_version='1',
        background_class=-1,
        max_output_boxes=topk,
        score_threshold=score_threshold, # 0.001  0.01
        iou_threshold=iou_threshold,
        score_activation=False,
        box_coding=0, )

    outputs = [
        onnx_graphsurgeon.Variable('num_dets', np.int32, [-1, 1]),
        onnx_graphsurgeon.Variable('det_boxes', np.float32, [-1, topk, 4]),
        onnx_graphsurgeon.Variable('det_scores', np.float32, [-1, topk]),
        onnx_graphsurgeon.Variable('det_classes', np.int32, [-1, topk])
    ]

    graph.layer(
        op='EfficientNMS_TRT',
        name="batched_nms",
        inputs=[graph.outputs[0], graph.outputs[1]],
        outputs=outputs,
        attrs=attrs)

    graph.outputs = outputs
    graph.cleanup().toposort()


    outputs =[node.name for node in graph.output]
    input_all = [node.name for node in graph.input]
    input_initializer =  [node.name for node in graph.initializer]
    inputs = list(set(input_all)  - set(input_initializer))
    print('inputs ', inputs)
    print('outputs ', outputs)

    onnx.save(onnx_graphsurgeon.export_onnx(graph), f'{name}_w_nms.onnx')


def to_binary_data(path):
    '''--loadInputs='image:input_tensor.dat'
    '''
    im = Image.open(path).resize((640, 640))
    data = np.asarray(im, dtype=np.float32).transpose(2, 0, 2)[None] / 255.
    data.tofile('input_tensor.dat')

    
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='yolov8l')
    parser.add_argument('--score_threshold', type=float, default=0.01)
    parser.add_argument('--iou_threshold', type=float, default=0.7)
    parser.add_argument('--max_output_boxes', type=int, default=300)

    args = parser.parse_args()

    export_onnx(name=args.name)
    onnx_insert_nms(name=args.name)
