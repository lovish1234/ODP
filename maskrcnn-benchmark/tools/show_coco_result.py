"""
python show_coco_result.py <result_dir>
"""
from pycocotools.coco import COCO
import torch
import sys
from skimage.io import imread
import matplotlib.pyplot as plt

res_dir = sys.argv[1]
root = "datasets/NYUv2/"
res_file = res_dir + "/segm.json"
index = 429
try:
    coco_res = torch.load(res_dir + "/coco_results.pth")
    print(coco_res)
except:
    print("Maskrcnn not installed!")

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