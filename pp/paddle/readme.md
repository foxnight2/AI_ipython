


```
https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html

cmake .. \
  -DWITH_MKL=ON \
  -DWITH_MKLDNN=ON \
  -DWITH_GPU=ON \
  -DWITH_DISTRIBUTE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH_NAME=Auto \
  -DPY_VERSION=3.7 \
  -DWITH_TENSORRT=ON \
  -DTENSORRT_ROOT=/paddle/software/TensorRT-8.5.1.7 \ 
  -DON_INFER=ON \
  -DWITH_TESTING=ON \

```