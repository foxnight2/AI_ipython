import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 0.9,
        height=ell_radius_y * 2.5,
        facecolor=facecolor,
        linestyle=(0, (5, 5)),
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y) - 0.1

    transf = transforms.Affine2D() \
        .rotate_deg(60) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

plt.figure(figsize=(6, 5))
plt.rcParams['font.sans-serif'] = ['Arial']
plt.title(r"MS COCO Object Detection", fontsize=18)
plt.xlabel('End-to-end Latency T4 TensorRT FP16 (ms)',fontsize=17)
plt.ylabel('COCO AP (%)',fontsize=17)
plt.axis([1, 27, 42.5, 57.4])
plt.xticks([4, 8, 12, 16, 20, 24], fontsize=18)
plt.yticks(fontsize=18)
model_infos = {
    'YOLOv5': {
        'Latency': [13.52, 15.55, 18.44, 23.14],
        'mAP': [37.4, 45.4, 49.0, 50.7],
        'Param': [46, 86],
        'marker': 's',
        'color': 'purple'
    },
    'PP-YOLOE': {
        'Latency': [4.59, 7.62, 10.67,16.80],
        'mAP': [43.0, 49.0, 51.4,52.3],
        'Param': [52, 98],
        'marker': 'o',
        'color': 'blue'
    },

    'YOLOv6-v3.0': {
        'Latency': [4.99, 8.28, 10.14],
        'mAP': [45.0, 50.0, 52.8],
        'Param': [59],
        'marker': '*',
        'color': 'c'
    },
    'YOLOv7': {
        'Latency': [18.14,22.32],
        'mAP': [51.2, 52.9],
        'Param': [36, 71],
        'marker': 'd',
        'color': 'green'
    },
    'YOLOv8': {
        'Latency': [7.37, 10.33, 14.0,20.10],
        'mAP': [44.9, 50.2, 52.9,53.9],
        'Param': [43, 68],
        'marker': 'v',
        'color': 'y'
    },
    
#     'RT-DETR(ours)': {
#             'Latency': [8.80,13.63],
#             'mAP': [53.0,54.8],
#             'Param': [32, 67],
#             'marker': 'h',
#             'color': 'red'
#         },
    'RT-DETR (ours)': {
            'Latency': [4.2, 4.6, 6.5, 6.9, 7.3, 7.6, 9.28, 13.58],
            'mAP': [45.5, 46.5, 50.3, 51.3, 51.8, 52.1, 53.1, 54.3],
            'Param': [42, 76],
            'marker': 'h',
            'color': 'red'
        },
    'RT-DETR-Obj365': {
        'Latency': [4.6, 9.28, 13.58],
        'mAP': [49.2, 55.3, 56.2],
        'Param': [42, 76],
        'marker': 'h',
        'color': 'red'
    },

    # 'PP-YOLOE+Obj365': {
    #     'Latency': [4.59, 7.62, 10.67,16.80],
    #     'mAP': [43.7, 49.8, 52.9, 54.7],
    #     'Param': [52, 98],
    #     'marker': 'o',
    #     'color': 'blue'
    # },

}
colors = ['b', 'c', 'r', 'y', 'm', 'g']
# shapes = ['*', 'v', 'x', '^', '<', '>', '+', 's', 'd', '.', '1', 'o', 'h']
shapes = ['*', '^', '+', 's', 'd', '.', 'h', 'p']

x_axis = 'Latency'

for name, info in model_infos.items():
    assert len(info[x_axis]) == len(info['mAP'])
    if 'Obj365' in name:
        # plt.plot(info[x_axis], info['mAP'], color=info['color'],
        #          marker=info['marker'], markersize=8, linewidth=1.0)
        plt.scatter(info[x_axis], info['mAP'], color=info['color'], marker=info['marker'], s=8**2, alpha=0.5)
    else:
        plt.plot(info[x_axis], info['mAP'], color=info['color'],
                marker=info['marker'], markersize=8, linewidth=1.0, label=name)
                
    if name == 'YOLOv5':
        labels = ['S', 'M', 'L', 'X']
        for i in range(len(info[x_axis])):
            if labels[i] == 'M':
                plt.text(model_infos['YOLOv5'][x_axis][i]-0.5, model_infos['YOLOv5']['mAP'][i]+0.3, labels[i], 
                         color='purple', fontsize=11, weight='bold')
            elif labels[i] == 'S':
                continue
            else:
                plt.text(model_infos['YOLOv5'][x_axis][i]-0.3, model_infos['YOLOv5']['mAP'][i]+0.3, labels[i], 
                         color='purple', fontsize=11, weight='bold')
    
    elif name == "PP-YOLOE":
        labels = ['S', 'M', 'L', 'X']
        for i in range(len(info[x_axis])):
            if labels[i] == 'M':
                plt.text(model_infos['PP-YOLOE'][x_axis][i]-0.6, model_infos['PP-YOLOE']['mAP'][i]+0.3, labels[i],
                         color='blue', fontsize=11, weight='bold')
            elif labels[i] == 'S':
                plt.text(model_infos['PP-YOLOE'][x_axis][i]-0.4, model_infos['PP-YOLOE']['mAP'][i]+0.3, labels[i],
                         color='blue', fontsize=11, weight='bold')
            else:
                plt.text(model_infos['PP-YOLOE'][x_axis][i]-0.35, model_infos['PP-YOLOE']['mAP'][i]+0.3, labels[i],
                         color='blue', fontsize=11, weight='bold')
    
    elif name == "YOLOv6-v3.0":
        labels = ['S', 'M', 'L']
        for i in range(len(info[x_axis])):
            if labels[i] == 'M':
                plt.text(model_infos['YOLOv6-v3.0'][x_axis][i]-0.6, model_infos['YOLOv6-v3.0']['mAP'][i]+0.3, labels[i],
                         color='c', fontsize=11, weight='bold')
            elif labels[i] == 'S':
                plt.text(model_infos['YOLOv6-v3.0'][x_axis][i]-0.30, model_infos['YOLOv6-v3.0']['mAP'][i]+0.25, labels[i],
                         color='c', fontsize=11, weight='bold')
            else:
                plt.text(model_infos['YOLOv6-v3.0'][x_axis][i]+0.15, model_infos['YOLOv6-v3.0']['mAP'][i]+0.13, labels[i],
                         color='c', fontsize=11, weight='bold')
            
    elif name == "YOLOv7":
        labels = ['L', 'X']
        for i in range(len(info[x_axis])):
            plt.text(model_infos['YOLOv7'][x_axis][i]-0.3, model_infos['YOLOv7']['mAP'][i]+0.4, labels[i],
                     color='green', fontsize=11, weight='bold')
    
    elif name == "YOLOv8":
        labels = ['S', 'M', 'L', 'X']
        for i in range(len(info[x_axis])):
            if labels[i] == 'M':
                plt.text(model_infos['YOLOv8'][x_axis][i]-0.4, model_infos['YOLOv8']['mAP'][i]+0.25, labels[i],
                         color='y', fontsize=11, weight='bold')
            elif labels[i] == 'S':
                plt.text(model_infos['YOLOv8'][x_axis][i]-0.4, model_infos['YOLOv8']['mAP'][i]+0.3, labels[i],
                         color='y', fontsize=11, weight='bold')
            else:
                plt.text(model_infos['YOLOv8'][x_axis][i]-0.3, model_infos['YOLOv8']['mAP'][i]+0.3, labels[i],
                         color='y', fontsize=11, weight='bold')

plt.text(model_infos['RT-DETR (ours)'][x_axis][0]-2.4, model_infos['RT-DETR (ours)']['mAP'][0]+1.0, 'R18', 
         color='red',fontsize=11,weight='bold')
plt.text(model_infos['RT-DETR (ours)'][x_axis][0]-3.0, model_infos['RT-DETR (ours)']['mAP'][0]+0.55, 'Scaled', 
         color='red',fontsize=11,weight='bold')
# plt.text(model_infos['RT-DETR (ours)'][x_axis][3]-2.4, model_infos['RT-DETR (ours)']['mAP'][3]+1.0, 'R34', 
#          color='red',fontsize=11,weight='bold')
# plt.text(model_infos['RT-DETR (ours)'][x_axis][3]-3.0, model_infos['RT-DETR (ours)']['mAP'][3]+0.55, 'Scaled', 
#          color='red',fontsize=11,weight='bold')
plt.text(model_infos['RT-DETR (ours)'][x_axis][4]-3.05, model_infos['RT-DETR (ours)']['mAP'][4], 'R50', 
         color='red',fontsize=11,weight='bold')
plt.text(model_infos['RT-DETR (ours)'][x_axis][4]-3.65, model_infos['RT-DETR (ours)']['mAP'][4]-0.45, 'Scaled', 
         color='red',fontsize=11,weight='bold')
plt.text(model_infos['RT-DETR (ours)'][x_axis][-2]-1.75, model_infos['RT-DETR (ours)']['mAP'][-2]+0.3, 'R50',
         color='red',fontsize=11,weight='bold')
plt.text(model_infos['RT-DETR (ours)'][x_axis][-1]-2.2, model_infos['RT-DETR (ours)']['mAP'][-1]+0.3, 'R101',
         color='red',fontsize=11,weight='bold')

plt.text(model_infos['RT-DETR-Obj365'][x_axis][0]-2.2, model_infos['RT-DETR-Obj365']['mAP'][0]+0.4, '+obj365',
         color='red',fontsize=11, weight='bold', alpha=0.5)
plt.text(model_infos['RT-DETR-Obj365'][x_axis][1]-1.9, model_infos['RT-DETR-Obj365']['mAP'][1]+0.4, '+obj365',
         color='red',fontsize=11, weight='bold', alpha=0.5)
plt.text(model_infos['RT-DETR-Obj365'][x_axis][2]-1.9, model_infos['RT-DETR-Obj365']['mAP'][2]+0.4, '+obj365',
         color='red',fontsize=11, weight='bold', alpha=0.5)



# bbox_R50
x = np.array([6.0, 6.8, 7.2, 8.0])
y = np.array([50.3, 51.3, 51.8, 52.1])
confidence_ellipse(x, y, plt.gca(), edgecolor='black')

# bbox_R18
x = np.array([3.8, 4.2, 4.4, 4.6, 5.0])
y = np.array([45.5, 45.5, 46.0, 46.5, 46.8])
confidence_ellipse(x, y, plt.gca(), edgecolor='black')

# +obj365
plt.arrow(4.6, 46.5, 0, 2.3, length_includes_head=True, head_width=0.4, head_length=1.3 * 0.4, fc='k', ec='k', alpha=0.5, linestyle='--', fill=False) # width=0.1, 
plt.arrow(9.28, 53.3, 0, 1.7, length_includes_head=True, head_width=0.4, head_length=1.3 * 0.4, fc='k', ec='k', alpha=0.5, linestyle='--', fill=False)
plt.arrow(13.58, 54.5, 0, 1.4, length_includes_head=True, head_width=0.4, head_length=1.3 * 0.4, fc='k', ec='k', alpha=0.5, linestyle='--', fill=False)


plt.grid(True, alpha=0.4)
plt.legend(loc='best', fontsize=10.5, edgecolor='black')
# plt.legend(loc=2, fontsize=10, bbox_to_anchor=(1.01, 1.02), ncol=1, borderaxespad=0, frameon=False)
# plt.subplots_adjust(right=0.6)
# plt.show()
plt.savefig("rtdetr_plot.pdf", format="pdf", bbox_inches='tight', dpi=400)
# plt.savefig("yoloseries_speed_plus.png")