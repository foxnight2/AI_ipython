

# https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
# https://github.com/pytorch/kineto/blob/main/tb_plugin/docs/gpu_utilization.md

# https://dev-discuss.pytorch.org/t/using-nsight-systems-to-profile-gpu-workload/59?u=ptrblck

import torch
from torch._C import channels_last, memory_format


class MM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.BatchNorm2d(32))

    def forward(self, x):
        return self.m(x)



with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU, 
        # torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/'),
    with_stack=True, 
    # with_flops=True,
    # profile_memory=True,
    # record_shapes=True,
    # with_modules=True,
) as p:

    mm = MM()
    a = torch.rand(1, 3, 640, 640)

    # mm = mm.to(memory_format=torch.channels_last)
    # a = a.to(memory_format=torch.channels_last)

    for i in range(10):
        b = mm(a)
        b.sum().backward()

        p.step()

    # print(p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))
    print(p.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))
    # print(p.key_averages().table())



