
# NV-TOOLS

- yum install perl-Env
- bash NsightSystems-linux-public-2021.2.1.58-642947b.run

```
/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -h 

/opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile --trace 'nvtx,cuda,osrt,cudnn' -c cudaProfilerApi -o test python deploy/python/infer.py --run_mode=trt_fp16 --device=GPU  --run_benchmark True --threshold=0.5 --output_dir=python_infer_output --image_dir=./demo --model_dir=output_inference/yolot_l_xhead_120e_coco/

```