

import json
from pycocotools.coco import COCO
import torch
import sys
from skimage.io import imread
import matplotlib.pyplot as plt
import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou


def prepare_for_coco_segmentation(predictions):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    coco_results = []

    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = image_id
        if len(prediction) == 0:
            continue

        image_width = 640
        image_height = 480
        # prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")

        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]

        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()


        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )

    return coco_results


"""
{'image_id': 0, 'category_id': 1.0, 'segmentation': {
    'size': [640, 480], 'counts': '\\T4k0i0WOD1N:c`0g0b_O`0?^Nf>c3k@oLf>[3o@jLo>W4OL3I60O \
    1OlNTATMl>]2fARMYO0Q?l2hASMXO2P?i2iATMXO3o>k2e12NN2O11O01N1O1100OO10010O0O101O2M5K8cMa^Oi1Qb0ROl]O@lTc8'}, 
    'score': 0.0507919117808342}
"""

res_dir = "demo/"
root = "datasets/ODPDemo/"
senario = ["rgb_baseline", "spade_no_depth", "spade_depth"]
all_predictions = None
demo_result = None
for s in senario:
    res_file = res_dir + s + "/inference/image_depth/segm.json"

    all_predictions  = torch.load(res_dir + s + "/inference/image_depth/predictions.pth")
    print(len(all_predictions))

    demo_result = prepare_for_coco_segmentation(all_predictions)
    print(len(demo_result))

    with open(res_dir + s + "_segm.json", 'w') as fout:
        json.dump(demo_result , fout)