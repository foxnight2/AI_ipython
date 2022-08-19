
# /opt/nvidia/nsight-systems/2021.2.1/bin/nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o model --capture-range=cudaProfilerApi --capture-range-end=stop --cudabacktrace=true --force-overwrite true --osrt-threshold=10000 -x true  
# python tools/train.py -c path/to/config.yml
# fleetrun 

# import paddle.fluid.core as core

start_step = 100
n_steps = 10
end_step = start_step + n_steps

# for step_id, data in enumerate(self.loader):
for step_id in range(len(self.loader)):

    if step_id == start_step:
        core.nvprof_start()
        core.nvprof_enable_record_event()
        core.nvprof_nvtx_push(str(step_id))
    if step_id == end_step:
        core.nvprof_nvtx_pop()
        core.nvprof_stop()
        sys.exit()
    if step_id > start_step and step_id < end_step:
        core.nvprof_nvtx_pop()
        core.nvprof_nvtx_push(str(step_id))


    # dataloader
    core.nvprof_nvtx_push('data')
    data = self.loader.next()
    core.nvprof_nvtx_pop()

    # model forward
    core.nvprof_nvtx_push('forward')
    outputs = model(data)
    core.nvprof_nvtx_pop()

    loss = outputs['loss']

    # model backward
    core.nvprof_nvtx_push('backward')
    loss.backward()
    core.nvprof_nvtx_pop()

    # optimizer
    core.nvprof_nvtx_push('optimizer')
    self.optimizer.step()
    self.optimizer.clear_grad()
    core.nvprof_nvtx_pop()


    # clear
    # core.nvprof_nvtx_push('clear_grad')
    # self.optimizer.clear_grad()
    # core.nvprof_nvtx_pop()


    # others
    core.nvprof_nvtx_push('others_GpuMemcpySync')
    # pass 
    core.nvprof_nvtx_pop()


    # ema
    core.nvprof_nvtx_push('ema')
    if self.use_ema:
        self.ema.update(self.model)
    core.nvprof_nvtx_pop()
    
    