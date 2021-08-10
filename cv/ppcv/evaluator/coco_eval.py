

# https://cocodataset.org/#detection-eval


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



annFile = '../../../coco/annotations/instances_train2017.json'
cocoGt=COCO(annFile)

resFile = './evaluator/demo_result_detection.txt'
cocoDt=cocoGt.loadRes(resFile)

# annType = ['segm','bbox','keypoints']

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

imgIds = sorted(cocoGt.getImgIds())
imgIds = imgIds[0:100]

cocoEval.params.imgIds  = imgIds

cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
