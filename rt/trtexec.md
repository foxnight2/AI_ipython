
```
../software/TensorRT-8.5.1.7/bin/trtexec --onnx=./yolov8l_w_nms.onnx --saveEngine=yolov8l_w_nms.engine --buildOnly --fp16

# --explicitBatch --minShapes=image:1x3x640x640 --optShapes=image:8x3x640x640  --maxShapes=image:16x3x640x640 --shapes=image:8x3x640x640


/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --force-overwrite=true  -t 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o yolov8l_w_nms  ../software/TensorRT-8.5.1.7/bin/trtexec --loadEngine=./yolov8l_w_nms.engine --fp16 --avgRuns=10 --loadInputs='image:in_data.bin'


# https://forums.developer.nvidia.com/t/about-loadinputs-in-trtexec/218880



```


/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --force-overwrite=true  -t 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o yoloel_w_nms  ../software/TensorRT-8.5.1.7/bin/trtexec --loadEngine=./yoloel_w_nms.engine --fp16 --avgRuns=10 --loadInputs='image:in_data.bin'