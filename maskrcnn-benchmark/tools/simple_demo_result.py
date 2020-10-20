"""
python show_demo_result.py <result_dir>
"""
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
        try:
            m = draw_bbox(m)
        except:
            pass
        img = np.ones( (m.shape[0], m.shape[1], 3) )
        cat_id = ann['category_id']
        color_mask = color_map[cat_id].tolist()
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack( (img, m*0.5) ))


def get_ann(image_id, anns):
    return [ann for ann in anns if ann['image_id'] == image_id and ann['score'] > 0.4]


"""
{'image_id': 0, 'category_id': 1.0, 'segmentation': {
    'size': [640, 480], 'counts': '\\T4k0i0WOD1N:c`0g0b_O`0?^Nf>c3k@oLf>[3o@jLo>W4OL3I60O \
    1OlNTATMl>]2fARMYO0Q?l2hASMXO2P?i2iATMXO3o>k2e12NN2O11O01N1O1100OO10010O0O101O2M5K8cMa^Oi1Qb0ROl]O@lTc8'}, 
    'score': 0.0507919117808342}
"""

keep_class = ["picture", "chair", "cabinet", "pillow", "paper", "table", "box", "books", "window", "lamp", "door", "bag", "shelves", "clothes", "sofa", "counter", "bed", "blinds", "desk", "mirror", "bookshelf", "curtain", "floor mat", "sink", "towel", "person"]

root = "datasets/ODPDemo/"
name = sys.argv[1]
segm_files = [f"{name}_segm.json"]
out_dirs = [name]

print("=> Loading images")
files = os.listdir(root + "image/")
files.sort()
images = [imread(root + "image/" + f) for f in files]

print("=> Visualizing")
for out_dir, segm_file in zip(out_dirs, segm_files):
    os.system(f"mkdir {root}/{out_dir}")
    segm = json.load(open(root + segm_file))

    for index, f in enumerate(files):
        image = imread(root + "image/" + f)
        ann_dt = get_ann(index, segm)
        plt.imshow(image); plt.axis('off')
        showAnns(ann_dt)
        plt.savefig(root + out_dir + ("/%04d.png" % index))
        plt.close()


"""
python simple_demo_result expr/ODPDemo/depth/inference/image_depth
"""       