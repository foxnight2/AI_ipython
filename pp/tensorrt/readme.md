
## TensorRT

- doc
  - https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable-fusion
  
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
  - pip install  TensorRT-7.2.3.4/python/xx.whl
  - export C_INCLUDE_PATH=/paddle/speed/TensorRT-8.2.2.1/include/:$C_INCLUDE_PATH
  - export LIBRARY_PATH=/paddle/speed/TensorRT-8.2.2.1/lib/:$LIBRARY_PATH
  <!-- - export CPLUS_INCLUDE_PATH=/paddle/speed/TensorRT-8.2.2.1/include/:$CPLUS_INCLUDE_PATH
  - export LD_LIBRARY_PATH=/paddle/speed/TensorRT-8.2.2.1/lib/:$LD_LIBRARY_PATH -->
  - /opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o yolo --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true --force-overwrite true --osrt-threshold=10000 -x true python tools/train.py


- docker
  - https://hub.docker.com/
  
  
## Pytorch
 
- torch2trt
  - https://github.com/NVIDIA-AI-IOT/torch2trt
  
 

## OpenCV
 - cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local  ..
 - make -j32
 - make install




python tools/trt.py -n yolox-s -c yolox_s.pth
python tools/trt.py -n yolox-m -c yolox_m.pth
python tools/trt.py -n yolox-l -c yolox_l.pth
python tools/trt.py -n yolox-x -c yolox_x.pth

sleep 20s
python -m yolox.tools.eval -n  yolox-s -b 1 -d 1 --conf 0.001 --trt

sleep 20s
python -m yolox.tools.eval -n  yolox-m -b 1 -d 1 --conf 0.001 --trt

sleep 20s
python -m yolox.tools.eval -n  yolox-l -b 1 -d 1 --conf 0.001 --trt

sleep 20s
python -m yolox.tools.eval -n  yolox-x -b 1 -d 1 --conf 0.001 --trt
