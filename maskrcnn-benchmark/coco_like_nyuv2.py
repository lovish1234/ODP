"""
Produce COCO like NYUV2
"""
import os
import torch
from torch.utils.data import Dataset
from os.path import join as osj
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
import datetime
from pycococreatortools import pycococreatortools

def imwrite(fpath, image):
    """
    image: np array, value range in [0, 255].
    """
    if ".jpg" in fpath or ".jpeg" in fpath:
        ext = "JPEG"
    elif ".png" in fpath:
        ext = "PNG"
    with open(osj(fpath), "wb") as f:
        Image.fromarray(image.astype("uint8")).save(f, format=ext)


def imread(fpath):
    with open(osj(fpath), "rb") as f:
        return np.asarray(Image.open(f))


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img


class NYUV2Dataset(Dataset):
    """
    Pytorch wrapper for NYUV2 dataset. Should use for training.
    """
    def __init__(
        self,
        ann_file: str,
        root: str,
        use_depth=False,
        debug=False,
        transforms=None
    ):
        super(NYUV2Dataset, self).__init__()
        self.root = root
        self.labeling = 40
        self.transforms = transforms
        self.use_depth = use_depth
        self.debug = debug
        self.depth_prob = 0.5 # The probability to drop depth

        self.split = "train" if "train" in ann_file else "test"

        #self.label40_data = np.load(
        #    self.root + "/labels40.npz",
        #    allow_pickle=True)['arr_0'][()]['labels40'].transpose(2, 0, 1)

        self.load_split()

        if self.debug:
            idx = np.load(osj(self.root, "nyuv2_splits.npy"), allow_pickle=True)[()]
            self.split_indice = {"train": idx["trainNdxs"], "test": idx["testNdxs"]}
            print("=> Root: %s" % self.root)
            print("=> Split: %s" % self.split)
            print(self.transforms)

    def load_split(self):
        self._files = os.listdir(osj(self.root, f"{self.split}_rgb"))
        self._files.sort()
        if self.use_depth:
            self.depth_data = np.load(
                self.root + f"/{self.split}_depth.npz",
                allow_pickle=True)['arr_0']
        with open(osj(self.root, f"{self.split}_seg{self.labeling}_annotations.json")) as fp:
            self.anno = json.load(fp)
            self.indice = list(self.anno.keys())

    def get_anno(self, index):
        return self.anno[self.indice[index]]

    def __getitem__(self, index: int):
        def folder(name):
            return osj(self.root, f"{self.split}_{name}")
        fname = self._files[index]
        anno = self.anno[self.indice[index]]

        image = pil_read(osj(folder("rgb"), fname))
        objects = imread(osj(folder("object"), fname.replace("nyu_rgb_", "")))
        n_obj = objects.max()
        masks = [(objects == i).astype("uint8")
            for i in range(1, n_obj + 1)]
        boxes = anno["bbox"]

        return fname,image, boxes, masks, anno['label'], self.indice[index]

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = "Dataset {self.__class__.__name__}\n"
        fmt_str += "    Number of data points: {self.__len__()}\n"
        fmt_str += "    Split: {self.split}\n"
        fmt_str += "    Root Location: {self.root}\n"
        return fmt_str

    def get_img_info(self, i):
        return {"height": 640, "width": 480}


INFO = {
    "description": "NYUV2 Dataset",
    "url": "",
    "version": "",
    "year": 0,
    "contributor": "Jianjin Xu",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

class13 = ["Bed","Books","Ceiling","Chair","Floor","Furniture","Object","Picture","Sofa","Table","TV","Wall","Window"]
class40 = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','blinds','desk','shelves','curtain','dresser','pillow','mirror','floor mat','clothes','ceiling','books','refridgerator','television','paper','towel','shower curtain','box','whiteboard','person','night stand','toilet','sink','lamp','bathtub','bag','otherstructure','otherfurniture','otherprop']
# select metric: number of instances > 100
keep_class = ["picture", "chair", "cabinet", "pillow", "paper", "table", "box", "books", "window", "lamp", "door", "bag", "shelves", "clothes", "sofa", "counter", "bed", "blinds", "desk", "mirror", "bookshelf", "curtain", "floor mat", "sink", "towel", "person"]
CATEGORIES = [
    {
        'id': i,
        'name': keep_class[i-1],
        'supercategory': keep_class[i-1],
    } for i in range(1, len(keep_class) + 1)
]

if __name__ == "__main__":
    rootdir = "datasets/NYUv2/"
    ds = NYUV2Dataset(rootdir + ("train_seg40_annotations.json"), rootdir)
    ds.labeling = 40
    for split in ['train', 'test']:
        ds.split = split
        ds.load_split()
        ann_id = 1
        img_id = 1
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": [],
            "annotations": []
        }
        for i, sample in enumerate(tqdm(ds)):
            fname, image, boxes, masks, labels, _ = sample
            image_fpath = split + "_rgb/" + fname
            image_info = pycococreatortools.create_image_info(
                img_id, image_fpath, image.size)
            coco_output["images"].append(image_info)
            for label, box, mask in zip(labels, boxes, masks):
                label_name = class40[label - 1]
                # ignore some classes
                if label == 0 or label_name not in keep_class:
                    continue
                label = keep_class.index(label_name) + 1
                category_info = {'id': label, 'is_crowd': 1}
                anno_info = pycococreatortools.create_annotation_info(
                        ann_id, img_id, category_info, mask,
                        image.size, tolerance=2)
                anno_info['width'] = image.size[0]
                anno_info['height'] = image.size[1]
                if anno_info is not None:
                    coco_output["annotations"].append(anno_info)
                ann_id += 1
            img_id += 1
        with open('{}/cocolike_nyuv2_{}_{}.json'.format(rootdir, len(keep_class), split), 'w') as f:
            json.dump(coco_output, f)

