"""
python show_coco_results.py <result_dir>
"""
from pycocotools.coco import COCO
import torch
import sys
from skimage.io import imread
import matplotlib.pyplot as plt
import os

root = "datasets/NYUv2/"
output_dir = "result/"
base_dir = "expr"

for scenario in os.listdir(base_dir):
    print (scenario)
    if '40' not in scenario.split("_") and 'inference' in os.listdir(os.path.join(base_dir,scenario)):
        res_dir = os.path.join(base_dir, scenario, 'inference', "nyuv2_depth_26_test")
        try:
            coco_res = torch.load(res_dir + "/coco_results.pth")
            print(coco_res)
        except:
            print("Maskrcnn not installed!")
        res_file = res_dir + "/segm.json"

        coco_gt = COCO(root + "cocolike_nyuv2_26_test.json")
        coco_dt = coco_gt.loadRes(res_file)

        if not os.path.exists(os.path.join(output_dir, scenario)):
            os.makedirs(os.path.join(output_dir, scenario))
        else:
            continue

        for i in range(1,100):
            image_data = coco_gt.loadImgs([i])[0]

            anns_gt = coco_gt.loadAnns(coco_gt.getAnnIds([i]))
            anns_dt = coco_dt.loadAnns(coco_dt.getAnnIds([i]))
            #anns_dt = [a for a in anns_dt if a['score'] > 0.5]
            
            image = imread(root + image_data['file_name'])

            plt.imshow(image); plt.axis('off')
            plt.savefig(os.path.join(output_dir,scenario,str(i)+"_input.png"))
            plt.close()

            plt.imshow(image); plt.axis('off')
            coco_gt.showAnns(anns_gt)
            plt.savefig(os.path.join(output_dir,scenario,str(i)+"_gt.png"))
            plt.close()

            plt.imshow(image); plt.axis('off')
            coco_dt.showAnns(anns_dt)
            plt.savefig(os.path.join(output_dir,scenario,str(i)+"_dt.png"))
            plt.close()
