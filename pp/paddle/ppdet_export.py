


# pip install gputil
# pip install nvidia-ml-py
# pip install pynvml


# python tools/export_model.py -c ./configs/faster_rcnn/faster_rcnn_swin_transformer_tiny_1x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_swin_tiny_fpn_1x_coco.pdparams TestReader.inputs_def.image_shape=[3,800,1333]


# python tools/export_model.py -c ./configs/faster_rcnn/faster_rcnn_swin_transformer_tiny_1x_coco.yml -o weights=https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_swin_tiny_fpn_1x_coco.pdparams


# python deploy/python/infer.py --model_dir=./output_inference/faster_rcnn_swin_transformer_tiny_1x_coco --image_file=./demo/000000014439_640x640.jpg --device=GPU --run_benchmark=True

# python deploy/python/infer.py --model_dir=./output_inference/faster_rcnn_swin_transformer_tiny_1x_coco --image_file=./demo/000000014439_640x640.jpg --device=GPU

# python deploy/python/infer.py --model_dir=./output_inference/faster_rcnn_swin_transformer_tiny_1x_coco/ --image_dir=./640/ --device=GPU --run_benchmark=True --run_mode=trt


# sh deploy/benchmark/benchmark.sh ./output_inference/faster_rcnn_swin_transformer_tiny_1x_coco faster_rcnn_swin_transformer_tiny_1x_coco




