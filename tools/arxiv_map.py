import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import rc
# rc('text', usetex=True)

plt.figure(figsize=(10, 5))
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.title(r"MS COCO Object Detection", fontsize=16)
plt.xlabel('FPS (V100)')
plt.ylabel('COCO mAP (%)')
plt.axis([36.5, 212.5, 36.5, 55.5,])
model_infos = {
    # 'EfficientDet': {
    #     'FPS': [98.0, 74.1, 56.5, 34.5],
    #     'mAP': [33.8, 39.6, 43.0, 45.8],
    #     'marker': 'o'
    # },
    'YOLOv5': {
        'FPS': [156.2, 121.9, 99.0, 82.6],
        'mAP': [37.4, 45.4, 49.0, 50.7],
        'marker': 'v',
        'color': 'purple'
    },
    'YOLOX': {
        'FPS': [119.0, 96.1, 62.5, 40.3],
        'mAP': [40.5, 47.2, 50.1, 51.5],
        'marker': 'o',
        'color': 'blue'
    },
    'PP-YOLO': {
        'FPS': [132.2, 109.6, 89.9, 72.9],
        'mAP': [39.3, 42.5, 44.4, 45.9],
        'marker': '*',
        'color': 'cyan'
    },
    'PP-YOLOv2': {
        # 'FPS': [123.3, 102.0, 93.4, 68.9, 50.3],
        # 'mAP': [43.1, 46.3, 48.2, 49.5, 50.3],
        'FPS': [68.9, 50.3],
        'mAP': [49.5, 50.3],
        'marker': 'd',
        'color': 'green'
    },
    'PP-YOLOE': {
        'FPS': [208.3, 123.4, 78.1, 45.0],
        'mAP': [43.1, 48.9, 51.4, 52.2],
        'marker': 's',
        'color': 'red'
    },
    'PP-YOLOE+': {
            'FPS': [208.3, 123.4, 78.1, 45.0],
            'mAP': [43.9, 50.0, 53.3, 54.9],
            'marker': 'h',
            'color': 'y'
        },
    # 'PP-YOLOE': {
    #         'FPS': [20.8, 20.2, 19.5, 18.5],
    #         'mAP': [43.0, 49.0, 51.4, 52.3],
    #         'marker': 's',
    #         'color': 'red'
    #     },
    # 'PP-YOLOE_plus': {
    #         'FPS': [47.8, 44.1, 42.2, 37.0],
    #         'mAP': [43.7, 49.8, 52.9, 54.7],
    #         'marker': 'o',
    #         'color': 'blue'
    #     },
}
colors = ['b', 'c', 'r', 'y', 'm', 'g']
# shapes = ['*', 'v', 'x', '^', '<', '>', '+', 's', 'd', '.', '1', 'o', 'h']
shapes = ['*', '^', '+', 's', 'd', '.', 'h', 'p']
for name, info in model_infos.items():
    plt.plot(info['FPS'], info['mAP'], color=info['color'],
             marker=info['marker'], markersize=8, linewidth=1.0, label=name)

# plt.text(3.0, 52.9, 'CBResNet', color='m')
# plt.text(21.2, 47.4, 'Cascade-Faster-RCNN', color='c')
# plt.text(68.9, 46.3, 'PP-YOLO', color='r')
# plt.text(62.6, 43.3, 'YOLOv3(ours)', color='r')
# plt.text(37.5, 32.6, 'YOLOv3(darknet)', color='r')
# plt.text(61., 33.4, 'EfficientDet-D0', color='b')
plt.grid(True, alpha=0.4)
plt.legend()
# plt.legend(loc=2, fontsize=10, bbox_to_anchor=(1.01, 1.02), ncol=1, borderaxespad=0, frameon=False)
plt.subplots_adjust(right=0.6)
# plt.show()
plt.savefig("ppyoloe_plus_map_fps.png", dpi=300)
