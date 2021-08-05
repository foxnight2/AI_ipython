import argparse


from kitti_eval_tools import kitti_common as kitti
from kitti_eval_tools.eval import kitti_eval

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

# from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
gt_split_file = "../../../kitti/ImageSets/val.txt"
val_image_ids = _read_imageset_file(gt_split_file)


gt_path = "../../../kitti/training/label_2/"
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

pred_path = "../../../kitti/training/label_2/"
pred_annos = kitti.get_label_annos(pred_path, val_image_ids, pred=True)


print(*kitti_eval(gt_annos, pred_annos, [0, 1])) # 6s in my computer

