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
from PIL import Image, ImageDraw, ImageFont


out_dirs = ['/work/OptionalDepthPathway/maskrcnn-benchmark/expr/ODPDemo/no_depth_demo']


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
    ret_x = xmin
    ret_y = ymin
    for delta in range(width):
        xmin += delta
        ymin += delta
        xmax -= delta
        ymax -= delta
        m[xmin:xmax, ymin] = m[xmin:xmax, ymax] = 1
        m[xmin, ymin:ymax] = m[xmax, ymin:ymax] = 1
    return m, ret_x, ret_y

color_map = labelcolormap(27)

def showAnns(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    color = []
    #count = 0
    for ann in anns:
        if type(ann['segmentation']['counts']) == list:
            rle = maskUtils.frPyObjects([ann['segmentation']], 640, 480)
        else:
            rle = [ann['segmentation']]
        m = maskUtils.decode(rle)
        xmin, ymin = 0,0
        try:
            m,xmin,ymin = draw_bbox(m)
        except:
            pass
        img = np.ones( (m.shape[0], m.shape[1], 3) )
        cat_id = ann['category_id']

        label = str(keep_class[cat_id])
        score = str(ann['score'])

        
        #print('## m SHAPE')
        #print(m.shape)
        color_mask = color_map[cat_id].tolist()
        for i in range(3):
            img[:,:,i] = color_mask[i]
        #print('## ORGINIAL')
        #print(img.shape)
        #print(np.amax(img))
        #print(np.amin(img))
        #img = Image.fromarray((img*255).astype('uint8'))

        #draw = ImageDraw.Draw(img)
        
        #fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 200)
        #draw.text((xmin,ymin), "test" , font = fnt, fill=(0,0,0,255))
        #draw.text((0,0), score, (255,255,0))
        #img.save(out_dir + ("/test%04d.png" % count))
        #count += 1
        #img = np.asarray(img)
        #img = img / 255
        #print('## TRANSFORM')
        #print(np.amax(img))
        #print(np.amin(img))
        #img = np.ones( (m.shape[0], m.shape[1], 3) ) * 0
        result = np.dstack( ((img*255).astype('uint8'), (m*255*0.5).astype('uint8')) )
        result = Image.fromarray(result,'RGBA')
        draw = ImageDraw.Draw(result)
        fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 13)
        draw.text((ymin,xmin), label + " " + score[:4] , font = fnt, fill=(0,0,0,255))
        result = np.asarray(result)
        #print('## DSTACK SHAPE')
        #print(result.shape)
        ax.imshow(result)


def get_ann(image_id, anns):
    return [ann for ann in anns if ann['image_id'] == image_id and ann['score'] > 0.4]


"""
{'image_id': 0, 'category_id': 1.0, 'segmentation': {
    'size': [640, 480], 'counts': '\\T4k0i0WOD1N:c`0g0b_O`0?^Nf>c3k@oLf>[3o@jLo>W4OL3I60O \
    1OlNTATMl>]2fARMYO0Q?l2hASMXO2P?i2iATMXO3o>k2e12NN2O11O01N1O1100OO10010O0O101O2M5K8cMa^Oi1Qb0ROl]O@lTc8'}, 
    'score': 0.0507919117808342}

python tools/simple_demo_result.py expr/ODPDemo/no_depth/inference/image_depth
    
"""

keep_class = ["picture", "chair", "cabinet", "pillow", "paper", "table", "box", "books", "window", "lamp", "door", "bag", "shelves", "clothes", "sofa", "counter", "bed", "blinds", "desk", "mirror", "bookshelf", "curtain", "floor mat", "sink", "towel", "person"]

root = "datasets/ODPDemo/"
name = sys.argv[1]
segm_files = [str(name) + "/segm.json"]

print("=> Loading images")
files = os.listdir(root + "image/")
files.sort()
images = [imread(root + "image/" + f) for f in files]

print("=> Visualizing")
for out_dir, segm_file in zip(out_dirs, segm_files):
    #os.system(f"mkdir {root}{out_dir}")
    segm = json.load(open(segm_file))
    print("## NUMBER OF IMAGES: " + str(len(files)))
    for index, f in enumerate(files):
        print("## " + str(index))
        image = imread(root + "image/" + f)
        ann_dt = get_ann(index, segm)
        plt.imshow(image); plt.axis('off')
        showAnns(ann_dt)
        plt.savefig(out_dir + ("/%04d.png" % index))
        plt.close()