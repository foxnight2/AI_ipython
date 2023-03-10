
# NV-TOOLS

- yum install perl-Env
- bash NsightSystems-linux-public-2021.2.1.58-642947b.run

```
/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -h 

/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --trace 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o test python deploy/python/infer.py --run_mode=trt_fp16 --device=GPU  --run_benchmark True --threshold=0.5 --output_dir=python_infer_output --image_dir=./demo --model_dir=output_inference/yolot_l_xhead_120e_coco/

```

## TensorRT
- download cudnn
  - tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda10-archive.tar.xz
  - cp include/* /usr/local/cuda/include/
  - cp lib/libcudnn* /usr/local/cuda/lib64/
  - chmod a+r /usr/local/cuda/include/cudnn.h 

- download tensort 
  - tar -xvf TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-10.2.cudnn8.6.tar.gz
  - export LD_LIBRARY_PATH=~/TensorRT-8.5.1.7/lib/:$LD_LIBRARY_PATH

- Nsight Systems
  - /opt/nvidia/nsight-systems/2021.2.1/bin/nsys  profile
  - --stats=True --force_overwrite -o state_report



<!-- /opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --help  -->

/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --stats=True --trace='nvtx,cuda,osrt,cudnn' --force_overwrite -o state_report

https://forums.developer.nvidia.com/t/about-loadinputs-in-trtexec/218880


## 测速
1. yolov5
```
a. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py#L305
b. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/engine/validator.py#L158
c. https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/utils/ops.py#L17


```


