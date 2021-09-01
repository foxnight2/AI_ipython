# ps aux | grep paddle_fleet.py | awk '{print $2}' | xargs kill -9
# fleetrun --gpus=0,1 --ips="10.21.226.186,10.181.196.11" paddle_fleet.py

# 10.21.226.186
# 10.181.196.11
# python -m paddle.distributed.launch --gpus 0,1 --ips="10.21.226.186,10.181.196.11" paddle_fleet.py
# python -m paddle.distributed.launch --gpus 0,1 --ips="10.181.196.11,10.21.226.186" paddle_fleet.py


import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed import fleet



class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear1 = nn.Linear(10, 10)
        self._linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self._linear2(self._linear1(x))

# 1. initialize fleet environment
# fleet.init(is_collective=True)
strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)

# 2. create layer & optimizer
layer = LinearNet()
loss_fn = nn.MSELoss()
adam = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=layer.parameters())

# 3. get data_parallel model using fleet
adam = fleet.distributed_optimizer(adam)
dp_layer = fleet.distributed_model(layer)

# 4. run layer
inputs = paddle.randn([10, 10], 'float32')
outputs = dp_layer(inputs)
labels = paddle.randn([10, 1], 'float32')
loss = loss_fn(outputs, labels)

print("loss:", loss.numpy())

loss.backward()

adam.step()
adam.clear_grad()
