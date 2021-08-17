
- Nsight Systems
  - https://developer.nvidia.com/gameworksdownload#?tx=$gameworks,developer_tools
  - bash
  - export PATH=/xxx/nvidia/nsight-systems/2021.2.1/bin/:$PATH
  - which nsys
  - CUDA_VISIBLE_DEVICES=0  nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o name_timeline --capture-range=cudaProfilerApi --stop-on-range-end=true --cudabacktrace=true --cudabacktrace-threshold=10000 --osrt-threshold=10000 -x true python -u tools/train.py -c configs/xxx.yml

  - [nvidia nsight systems](https://developer.nvidia.com/nsight-systems)
  
  

```
from paddle.fulid import core 

_N = 500
_M = _N + 10

if iter_id == _N:
    core.nvprof_start()
    core.nvprof_enable_record_event()
    core.nvprof_nvtx_push(str(iter_id))
if iter_id == _M :
    core.nvprof_nvtx_pop()
    core.nvprof_stop()
    import sys
    sys.exit()
if _M > iter_id > _N:
    core.nvprof_nvtx_pop()
    core.nvprof_nvtx_push(str(iter_id))
```