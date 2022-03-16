import timm
import torch

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


print( timm.list_models('resnet*') )


model = timm.create_model('resnet18', pretrained=False)

print(model)
print(model.default_cfg)



train_nodes, eval_nodes = get_graph_node_names(model)
print(train_nodes)


features = {'act1': 'feat1', 'layer4.1.act2': 'feat2'}


data = torch.rand(1, 3, 320, 320)
mm = create_feature_extractor(model, features)

for k, v in mm(data).items():
    print(k, v.shape)




# CSPResNetB_silu_large_stem_last_conv	base+multi-scale+mixup（timm）