"""
python show_demo_result.py <> <segm_json_dir>
"""

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

import os
import json
from pycocotools.coco import COCO
import torch
import sys
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import numpy as np
from skimage.io import imread

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r / 255.
        cmap[i, 1] = g / 255.
        cmap[i, 2] = b / 255.
    return cmap


def draw_bbox(m, width=2):
    # Add (psedo) bounding box
    w = np.where(m.max(0))[0]
    h = np.where(m.max(1))[0]
    ymin, ymax = w.min(), w.max()
    xmin, xmax = h.min(), h.max()
    for delta in range(width):
        xmin += delta
        ymin += delta
        xmax -= delta
        ymax -= delta
        m[xmin:xmax, ymin] = m[xmin:xmax, ymax] = 1
        m[xmin, ymin:ymax] = m[xmax, ymin:ymax] = 1
    return m

color_map = labelcolormap(27)

def showAnns(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    color = []
    for ann in anns:
        if type(ann['segmentation']['counts']) == list:
            rle = maskUtils.frPyObjects([ann['segmentation']], 640, 480)
        else:
            rle = [ann['segmentation']]
        m = maskUtils.decode(rle)
        m = draw_bbox(m)
        img = np.ones( (m.shape[0], m.shape[1], 3) )
        cat_id = ann['category_id']
        color_mask = color_map[cat_id].tolist()
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack( (img, m*0.5) ))


def get_ann(image_id, anns):
    return [ann for ann in anns if ann['image_id'] == image_id]



"""
{'image_id': 0, 'category_id': 1.0, 'segmentation': {
    'size': [640, 480], 'counts': '\\T4k0i0WOD1N:c`0g0b_O`0?^Nf>c3k@oLf>[3o@jLo>W4OL3I60O \
    1OlNTATMl>]2fARMYO0Q?l2hASMXO2P?i2iATMXO3o>k2e12NN2O11O01N1O1100OO10010O0O101O2M5K8cMa^Oi1Qb0ROl]O@lTc8'}, 
    'score': 0.0507919117808342}
"""


res_dir = sys.argv[1]
root = "datasets/ODPDemo/"
res_file = res_dir + "/segm.json"
save_dir = "datasets/ODPDemo/spade_depth"

#senario = [spade_no_depth", "spade_depth"]



coco_gt = COCO("datasets/NYUv2/cocolike_nyuv2_26_test.json")
coco_dt = coco_gt.loadRes(res_file)

keep_class = ["picture", "chair", "cabinet", "pillow", "paper", "table", "box", "books", "window", "lamp", "door", "bag", "shelves", "clothes", "sofa", "counter", "bed", "blinds", "desk", "mirror", "bookshelf", "curtain", "floor mat", "sink", "towel", "person"]

for index in range(1150):

    index += 1
    anns_dt_COCO = coco_dt.loadAnns(coco_dt.getAnnIds([index]))
    anns_dt_COCO = [a for a in anns_dt_COCO if a['score'] > 0.5]


    #print("image " + str(index) + " has " + str(len(anns_dt_COCO)) + " masks")
    

    img_name = str(index)
    while len(img_name) < 4:
        img_name = "0" + img_name
    img_name = img_name + ".png"    
    #print(img_name)

    image = imread(root + "image/" + img_name)
    #print(image.shape)


    plt.imshow(image); plt.axis('off')
    showAnns(anns_dt_COCO)
    #coco_dt.showAnns(anns_dt_COCO)
    plt.savefig(save_dir  + "/"  + img_name + ".png")
    plt.close()

"""
try:
    coco_res = torch.load(res_dir + "/predictions.pth")
    print(coco_res)
except:
    print("Maskrcnn not installed!")
"""


"""
coco_gt = COCO(root + "cocolike_nyuv2_26_test.json")
coco_dt = coco_gt.loadRes(res_file)
image_data = coco_gt.loadImgs([index])[0]
anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds([index]))
anns_dt = coco_dt.loadAnns(coco_dt.getAnnIds([index]))
anns_dt = [a for a in anns_dt if a['score'] > 0.5]
image = imread(root + image_data['file_name'])

plt.imshow(image); plt.axis('off')
plt.savefig("input.png")
plt.close()

plt.imshow(image); plt.axis('off')
coco_gt.showAnns(anns_gt)
plt.savefig("gt.png")
plt.close()

plt.imshow(image); plt.axis('off')
coco_dt.showAnns(anns_dt)
plt.savefig("dt.png")
plt.close()
"""