import os
import argparse
from os.path import join as osj
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from collections import OrderedDict

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results

argparser = argparse.ArgumentParser()
argparser.add_argument("--compute", type=int, default=0, help="Compute the result from prediction or load from computed result.")
args = argparser.parse_args()

basedir = "results/spade"
exprs = os.listdir(basedir)
exprs.sort()
cases = ["depth", "no_depth"]
splits = ["test"]
res_files = ["bbox.json", "segm.json"]
root = "datasets/NYUv2/"
strs = ""

def args_gen():
    for expr in exprs:
        for c in cases:
            for sp in splits:
                for model in os.listdir(osj(basedir, expr, c, sp)):
                    yield expr, c, sp, model

if args.compute:
    coco_gt = COCO(root + "cocolike_nyuv2_26_test.json")
    for expr, c, sp in args_gen():
        strs += f"\n{expr}_{c}_{sp}"
        coco_results = COCOResults("bbox", "segm")
        for r in res_files:
            res_file = osj(basedir, expr, c, sp, r)
            if not os.path.exists(res_file):
                continue
            coco_dt = coco_gt.loadRes(res_file)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType=r.replace(".json", ""))
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            coco_results.update(coco_eval)
        strs += str(coco_results)
    strs = strs.replace(",", " &")
    print(expr)
    print(strs)
    with open(osj(basedir, "result.txt"), "w") as f:
        f.write(strs)
else:
    import torch
    for expr, c, sp, model in args_gen():
        strs += f"\n{expr}_{c}_{sp}_{model}"
        res_file = osj(basedir, expr, c, sp, model, "coco_results.pth")
        if not os.path.exists(res_file):
            continue
        res = torch.load(res_file)
        strs += str(res)
    strs = strs.replace(",", " &")
    print(strs)
    with open(osj(basedir, "result.txt"), "w") as f:
        f.write(strs)

"""
Depth:
Task: bbox
AP, AP50, AP75, APs, APm, APl \\
0.3191, 0.4400, 0.3370, 0.0783, 0.2138, 0.4506 \\
Task: segm
AP, AP50, AP75, APs, APm, APl \\
0.2866, 0.4061, 0.3006, 0.0276, 0.1549, 0.4725 \\

No depth:
Task: bbox
AP, AP50, AP75, APs, APm, APl \\
0.2624, 0.3781, 0.2792, 0.0671, 0.1637, 0.3813 \\
Task: segm
AP, AP50, AP75, APs, APm, APl \\
0.2320, 0.3386, 0.2428, 0.0204, 0.1141, 0.4028 \\
"""