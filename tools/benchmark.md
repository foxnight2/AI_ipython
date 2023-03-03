


```
name=yolov5

python generate_yaml.py --input_file=./Models/${name}/infer_input.txt --yaml_file=./Models/${name}/infer_input.yaml 

echo "-----tensorrt-----"
sleep 10s

python benchmark.py --model_dir=Models/${name}/ --config_file infer_input.yaml --backend_type=tensorrt --batch_size=1 --enable_gpu=true --gpu_id=0 --enable_trt=true --precision=fp16

echo "-----paddle-trt-----"
sleep 10s

python benchmark.py --model_dir=Models/${name}/ --config_file infer_input.yaml --backend_type=paddle --batch_size=1 --enable_gpu=true --gpu_id=0 --enable_trt=true --precision=fp16 --paddle_model_file "model.pdmodel" --paddle_params_file "model.pdiparams"

echo "-----trtexec------"
sleep 10s

/paddle/software/TensorRT-8.5.1.7/bin/trtexec --onnx=./Models/${name}/model.onnx --workspace=4096 --avgRuns=1000 --shapes=image:1x3x640x640,scale_factor:1x2 --fp16

```


