
## TensorRT

- cuda
  - nvcc -V
  - cat /usr/local/cuda/version.txt
  - https://developer.nvidia.com/cuda-toolkit-archive
  
- cudnn
  - cat /usr/local/cuda/include/cudnn.h 
  - https://developer.nvidia.com/rdp/cudnn-archive
  
- tensorrt
  - https://developer.nvidia.com/nvidia-tensorrt-download
  - https://docs.nvidia.com/deeplearning/tensorrt/install-guide
  - tar xzvf TensorRT-7.2.3.4.Ubuntu-18.04.4.x86_64-gnu.cuda-11.2.cudnn7.3.tar
  - export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-7.2.3.4/lib
  - pip install  xx/python/xx.whl
  
- docker
  - https://hub.docker.com/
  
  
## Pytorch
 
- torch2trt
  - https://github.com/NVIDIA-AI-IOT/torch2trt
  
 