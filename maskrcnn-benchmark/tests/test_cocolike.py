from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import pylab

image_directory = 'datasets/NYUv2/'
annotation_file = image_directory + '/cocolike_nyuv2_26_train.json'
example_coco = COCO(annotation_file)

categories = example_coco.loadCats(example_coco.getCatIds())
print(categories)
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))

category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['bed'])
image_ids = example_coco.getImgIds(catIds=category_ids)
image_data = example_coco.loadImgs([image_ids[np.random.randint(0, len(image_ids))]])[0]
print(image_data)

image = io.imread(image_directory + image_data['file_name'])
plt.imshow(image); plt.axis('off')
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'])
annotations = example_coco.loadAnns(annotation_ids)
example_coco.showAnns(annotations)
plt.savefig("temp.png")
plt.close()