from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
import seaborn as sns
sns.set()

def compute_statistic(root="datasets/NYUv2/", ann_file="cocolike_nyuv2_13_train.json", classes=40):
    coco = COCO(root + ann_file)
    cat_ids = coco.getCatIds()

    name = [obj['name'] for obj in coco.cats.values()]
    count = []
    area = []
    for label in cat_ids:
        anns = coco.loadAnns(coco.getAnnIds(catIds=[label]))
        count.append(len(anns))
        area.append([obj['area'] for obj in anns])

    name_count = sorted(list(zip(count, name)), reverse=True)

    
    ax = plt.subplot(1,1,1)
    ax.bar(
            [n[1] for n in name_count][0:classes],
            [n[0] for n in name_count][0:classes])
    ax.set_xticklabels([n[1] for n in name_count][0:classes], rotation=90)

    [t.set_color(i) for (i,t) in zip(['red', 'red', 'red'], [ax.xaxis.get_ticklabels()[0], ax.xaxis.get_ticklabels()[5], ax.xaxis.get_ticklabels()[8]])]
        
    plt.savefig("class_frequency_40.png", bbox_inches='tight')
    plt.close()

if __name__=='__main__':
    classes=40
    ann_file = "cocolike_nyuv2_"+str(classes)+"_train.json"
    compute_statistic(ann_file=ann_file)
