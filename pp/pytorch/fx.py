import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.fx
from torch.fx import symbolic_trace



class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 8)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

net = Net()
# print(net)

net_traced = symbolic_trace(net)
print(net_traced.graph)
print(net_traced.code)
print(net_traced.graph.print_tabular())


def quant_and_eval():
    import copy
    import time 
    import torchvision
    from torch.quantization import get_default_qconfig
    from torch.quantization.quantize_fx import prepare_fx, convert_fx

    m = torchvision.models.resnet18()
    m.eval()

    qconfig = get_default_qconfig(backend='fbgemm')
    qconfig_dict = {
        "": qconfig,
    }

    _m = copy.deepcopy(m)
    m_prepared = prepare_fx(_m, qconfig_dict=qconfig_dict)
    m_int8 = convert_fx(m_prepared, )

    m_int8.load_state_dict(m_int8.state_dict())
    m_int8.eval()

    for k in m_int8.state_dict():
        print(k)

    x = torch.rand(1, 3, 224, 224)

    t0 = time.time()

    for _ in range(10):
        out1 = m(x)

    t1 = time.time()
    print(t1 - t0)

    for _ in range(10):
        out2 = m_int8(x)

    t2 = time.time()
    print(t2 - t1)


    print(torch.argmax(out1))
    print(torch.argmax(out2))


    # test_loader: DataLoader
    # evaluate_model(model, test_loader)
    # evaluate_model(model_int8, test_loader)

    # calib
    # def calib_quant_model(model, dataloader):
    #     model.eval()
    #     with torch.inference_mode():
    #         for inputs, labels in dataloader:
    #             _ = model(inputs)
    #     print('calib done')
    
    # calib_quant_model(test_loader)


    # torch.onnx.export(m_int8, x, 'm_int8.onnx')


if __name__ == '__main__':


    x = torch.rand(1, 3, 224, 224)
    quant_and_eval()
