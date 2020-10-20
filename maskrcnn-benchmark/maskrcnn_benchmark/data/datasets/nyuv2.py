"""
author: Jianjin Xu
date: 20/10/2019
"""
import os
import json
import h5py
import torch
import torchvision
import numpy as np
from PIL import Image
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import Dataset
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.bounding_box import BoxList
from os.path import join as osj

def mat2npy(fpath):
	"""
	For NYUv2:
		mat2npy("class13Mapping.mat","class13Mapping")
		mat2npy("classMapping40.mat","className")
	"""
	import scipy.io
	mat = scipy.io.loadmat(fpath)
	dic = {}
	for key in mat.keys():
		if "__" not in key:
			print(key)
			dic.update({key:mat[key]})
	np.save(fpath.replace(".mat", ".npy"), dic)


def imread(fpath):
    with open(osj(fpath), "rb") as f:
        return np.asarray(Image.open(f))


def pil_read(fpath):
    with open(os.path.join(fpath), "rb") as f:
        img = Image.open(f)
        img.load()
    return img


class COCOLikeNYUV2Dataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root,
        use_depth=True, use_estimated_depth=False, use_hha=False,
        debug=False, transforms=None
    ):
        super(COCOLikeNYUV2Dataset, self).__init__(root, ann_file)
        self.use_depth = use_depth
        self.use_estimated_depth = use_estimated_depth
        self.use_hha = use_hha
        self.debug = debug
        self.depth_prob = 0.5
        self.split = "train" if "train" in ann_file else "test"

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms
        self.load_split()
        self.rng = np.random.RandomState(1314)

    def load_split(self):
        if self.use_depth:
            self.depth_max = 9.99547
            self.depth_min = 0.7132995
            self.depth_data = np.load(
                f"{self.root}/{self.split}_depth.npz",
                allow_pickle=True)['arr_0']
            self.depth_data = (self.depth_data - self.depth_min) / (self.depth_max - self.depth_min) * 2 - 1
        if self.use_hha:
            self.hha_dir = f"{self.root}/{self.split}_hha"
        if self.use_estimated_depth:
            self.est_depth_data = np.load(
                f"{self.root}/{self.split}_depth_estimation_densedepth.npz",
                allow_pickle=True)['arr_0']
            depth_max = np.amax(self.est_depth_data)  
            depth_min = np.amin(self.est_depth_data)  
            self.est_depth_data = (self.est_depth_data - depth_min) / (depth_max - depth_min) * 2 - 1    

    def __getitem__(self, idx):
        while True:
            img, anno = super(COCOLikeNYUV2Dataset, self).__getitem__(idx)
            if len(anno) > 0 or self.split == "test":
                break
            idx = np.random.randint(0, len(self))

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            bin_masks = [torch.from_numpy(self.coco.annToMask(obj))
                for obj in anno]
            masks = SegmentationMask(bin_masks, img.size, mode='mask')
            target.add_field("masks", masks)

        target = target.clip_to_image(remove_empty=True)

        if self.use_depth:
            depth = torch.from_numpy(self.depth_data[idx]).unsqueeze(0)
            if self.use_estimated_depth:
                est_depth = torch.from_numpy(
                    self.est_depth_data[idx]).unsqueeze(0)
                depth = torch.cat([depth, est_depth], 0) # cat est depth as 5th channel
                #print("cat depth and est in nyuv2.py")
                #print(depth.shape)
            if self.use_hha:
                hha = torch.from_numpy(
                    imread("{}/{}.png".format(self.hha_dir, idx + 1)))
                hha /= 100
                # DHHA representation
                depth = torch.cat([depth, hha], 0)
            
            image, target, depth_ = self._transforms(img, target, depth)
            image = torch.cat([image, depth_], 0)
        else:
            image, target = self._transforms(img, target)

        if self.debug:
            image_np = np.asarray(img).copy()
            print("Image %s %f %f" % (str(image_np.shape), image_np.min(), image_np.max()))
            if self.use_depth:
                print("Depth %s %f %f" % (str(depth.shape), depth.min(), depth.max()))
                depth_np = depth.detach().cpu().numpy()
                depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
                depth_np = (depth_np * 255).astype("uint8")
                imwrite("%04d_d.png" % idx, depth_np)
            imwrite("%04d.png" % idx, image_np)
            for i in range(len(target)):
                box = [int(b) for b in boxes[i]] # xmin, ymin, xmax, ymax
                box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                mask = bin_masks[i].unsqueeze(2).numpy()
                image_np[box[1],box[0]:box[2],:] = 255 - i * 10, 0, 0
                image_np[box[3],box[0]:box[2],:] = 255 - i * 10, 0, 0
                image_np[box[1]:box[3],box[0],:] = 255 - i * 10, 0, 0
                image_np[box[1]:box[3],box[2],:] = 255 - i * 10, 0, 0
                box_image = (image_np * mask)[box[1]:box[3], box[0]:box[2], :]
                print(box, box_image.shape, image_np.shape, mask.shape)
                print(box_image.max(), mask.max())
                imwrite("%04d_%02d.png" % (idx, i), box_image)
                imwrite("%04d_%02d_.png" % (idx, i), image_np * mask)
            imwrite("%04d_.png" % idx, image_np)

        #print(image.shape, image[:3].max(), image[:3].min(), image[3].max(), image[3].min())

        return image, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data


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
        self.transforms = transforms
        self.use_depth = use_depth
        self.debug = debug
        self.depth_prob = 0.5 # The probability to drop depth

        self.split = "train" if "train" in ann_file else "test"

        self.label40_data = np.load(
            self.root + "/labels40.npz",
            allow_pickle=True)['arr_0'][()]['labels40'].transpose(2, 0, 1)
        idx = np.load(osj(self.root, "nyuv2_splits.npy"), allow_pickle=True)[()]
        self.split_indice = {"train": idx["trainNdxs"], "test": idx["testNdxs"]}

        self.load_split()

        if self.debug:
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
        with open(osj(self.root, f"{self.split}_annotations.json")) as fp:
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
        masks = [torch.from_numpy((objects == i).astype("int"))
            for i in range(1, n_obj + 1)]

        # in default image is concatenated into image
        ### TEST: no depth, no transform
        #image_t = torch.from_numpy(image).permute(3, 0, 1)

        boxes = torch.as_tensor(anno["bbox"]).reshape(-1, 4)
        target = BoxList(boxes, image.size, mode="xyxy")
        target.add_field("labels", torch.tensor(anno["label"]))
        target.add_field("masks", SegmentationMask(masks, image.size, "mask"))

        if self.use_depth:
            if np.random.rand() < self.depth_prob:
                depth = torch.zeros(objects.shape[0], objects.shape[1]).float()
            else:
                depth = torch.from_numpy(self.depth_data[index])
            image_, target_, depth_ = self.transforms(image, target, depth)
            image_ = torch.cat([image_, depth_], 0)
        else:
            image_, target_ = self.transforms(image, target)

        if self.debug:
            idx = self.split_indice[self.split][index]
            print(self.indice[index], fname)
            print("Image %s %f %f" % (str(image_.shape), image_[:3].min(), image_[:3].max()))
            if self.use_depth:
                print("Depth %f %f" % (image_[3].min(), image_[3].max()))
            print("Mask", masks[0].shape, masks[0].min(), masks[0].max())
            image_np = np.asarray(image).copy()
            imwrite("%04d.png" % idx, image_np)
            for i in range(n_obj):
                box = anno["bbox"][i] # xmin, ymin, xmax, ymax
                mask = masks[i].unsqueeze(2).numpy()
                image_np[box[1],box[0]:box[2],:] = 255,0,0
                image_np[box[3],box[0]:box[2],:] = 255,0,0
                image_np[box[1]:box[3],box[0],:] = 255,0,0
                image_np[box[1]:box[3],box[2],:] = 255,0,0
                box_image = (image_np * mask)[box[1]:box[3], box[0]:box[2], :]
                print(box, box_image.shape, image_np.shape, mask.shape)
                print(box_image.max(), mask.max())
                imwrite("%04d_%02d.png" % (idx, i), box_image)
                imwrite("%04d_%02d_.png" % (idx, i), image_np * mask)
            imwrite("%04d_.png" % idx, image_np)

        return image_, target_, index

    def __len__(self):
        return len(self._files)

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self.split}\n"
        fmt_str += f"    Root Location: {self.root}\n"
        return fmt_str

    def get_img_info(self, i):
        return {"height": 480, "width": 640}
        

class MatNYUv2(Dataset):
    """
    Read data from NYU mat file. Used to extract instance data, not for training.
    """
    def __init__(self, root_dir="datasets/NYUv2", is_train=True):
        self.root_dir = root_dir
        self.mat_fpath = osj(root_dir, "nyu_depth_v2_labeled.mat")
        self.mat_file = h5py.File(self.mat_fpath, 'r')
        self.split = "train" if is_train else "test"
        idx = np.load(osj(root_dir, "nyuv2_splits.npy"), allow_pickle=True)[()]
        self.split_indice = {"train": idx["trainNdxs"], "test": idx["testNdxs"]}

        self.depth_data = np.asarray(self.mat_file['depths'])
        self.image_data = np.asarray(self.mat_file['images'])
        self.label_data = np.asarray(self.mat_file['labels'])
        self.instance_data = np.asarray(self.mat_file['instances'])
	
    def __len__(self):
        return len(self.split_indice[self.split])

    def __getitem__(self, idx):
        idx = self.split_indice[self.split][idx] - 1
        image = self.image_data[idx]
        depth = self.depth_data[idx]
        instance = self.instance_data[idx]
        label = self.label_data[idx]
        return image, depth, instance, label, idx + 1

    def __repr__(self):
        fmt_str = f"Dataset {self.__class__.__name__}\n"
        fmt_str += f"    Number of data points: {self.__len__()}\n"
        fmt_str += f"    Split: {self.split}\n"
        fmt_str += f"    Root Location: {self.root_dir}\n"
        return fmt_str


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        if len(size) == 2:
            h, w = size
        else:
            h, w = size[1:]
            gray_image = gray_image[0]
        color_image = torch.ByteTensor(3, h, w).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    """
    edge_num = int(np.ceil(np.power(N + , 1/3))) - 1
    cmap = np.zeros((N, 3), dtype=np.uint8)
    step_size = 255. / edge_num
    cmap[0] = (0, 0, 0)
    count = 1
    for i in range(edge_num + 1):
        for j in range(edge_num + 1):
            for k in range(edge_num + 1):
                if count >= N or (i == j and j == k):
                    continue
                cmap[count] = [int(step_size * n) for n in [i, j, k]]
                count += 1
    """

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7-j))
            g = g ^ (np.uint8(str_id[-2]) << (7-j))
            b = b ^ (np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def numpy2label(label_np, n_label):
    img_t = Colorize(n_label)(torch.from_numpy(label_np))
    return img_t.numpy().transpose(1, 2, 0)


def get_instance_mask(segmentation, instance):
    # compute instance score
    def inst_score(inst, seg):
        inst_in_seg = float((inst & seg).sum()) / inst.sum()
        seg_in_inst = float((inst & seg).sum()) / seg.sum()
        return max(inst_in_seg, seg_in_inst)

    # not consider 0: this is not annotated class
    seg_label = np.unique(segmentation)[1:]
    segs = np.zeros((seg_label.shape[0], segmentation.shape[0], segmentation.shape[1]))
    # not consider 0: this is not annotated class
    inst_label = np.unique(instance)[1:]
    instances = np.zeros((inst_label.shape[0], instance.shape[0], instance.shape[1]))
    objects = []

    for i, s in enumerate(seg_label):
        segs[i] = (segmentation == s)
        #imwrite("seg_%d.png" % i, segs[i].astype("uint8") * 255)
    
    for i, inst in enumerate(inst_label):
        instances[i] = (instance == inst)
        #imwrite("inst_%d.png" % i, instances[i].astype("uint8") * 255)
    
    segs = segs.astype("bool")
    instances = instances.astype("bool")

    # match the instance and label
    while True:
        pos = []
        score = []
        for i in range(seg_label.shape[0]):
            if segs[i].sum() < 1:
                continue
            for j in range(inst_label.shape[0]):
                if instances[j].sum() < 1:
                    continue

                score.append(inst_score(instances[j], segs[i]))
                pos.append((i, j))

        if len(score) == 0:
            break

        ind = np.argmax(score)
        i, j = pos[ind]

        mask = (segs[i] & instances[j])
        instances[j] = instances[j] & (~mask)
        segs[i] = segs[i] & (~mask)
        mask = mask.astype("float32")
        objects.append(mask * seg_label[i])

    for i in range(seg_label.shape[0]):
        if segs[i].sum() > 1:
            print("segmentation left over")
            imwrite("seg_left_%d.png" % i,
                segs[i].astype("uint8") * 255)
    
    for i in range(inst_label.shape[0]):
        if instances[i].sum() > 1:
            print("instance left over")
            imwrite("inst_left_%d.png" % i,
                instances[i].astype("uint8") * 255)
    
    return np.array(objects)

# The object's number is default label, not meant to be read.
def get_object_label_and_box(objects, segmentation):
    n_obj = objects.max()
    label = np.unique(segmentation)
    matched = np.zeros((n_obj,), dtype="bool")
    obj_label = [0] * n_obj
    bboxes = [0] * n_obj
    for l in label:
        mask = (segmentation == l)
        for j in range(n_obj):
            if matched[j]:
                continue
            obj = (objects == (j + 1))
            w = np.where(obj.max(0))[0]
            h = np.where(obj.max(1))[0]
            xmin, xmax = w.min(), w.max()
            ymin, ymax = h.min(), h.max()

            if (obj & mask == obj).all():
                matched[j] = True
                obj_label[j] = int(l)
                bboxes[j] = [int(xmin), int(ymin), int(xmax), int(ymax)]
    assert (matched == True).all()
    return obj_label, bboxes


## Test helper function


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


if __name__ == "__main__":
    rootdir = "datasets/NYUv2"
    from maskrcnn_benchmark.data.HHA.compute import getHHA
    depth = np.load("datasets/NYUv2/test_depth.npz")["arr_0"]

    # generate HHA
    

    """ # colorize seg13
    for split in ["train", "test"]:
        out_dir = "%s/%s_seg13_viz" % (rootdir, split)
        in_dir = "%s/%s_seg13" % (rootdir, split)
        os.system("rm -r %s" % out_dir)
        os.system("mkdir %s" % out_dir)
        files = os.listdir(in_dir)
        files.sort()
        for f in files:
            fpath = in_dir + "/" + f
            label = imread(fpath)
            imwrite("%s/%s" % (out_dir, f), numpy2label(label, 14))
    """

    """ # get bounding box
    labeling = 40
    idx = np.load(osj(rootdir, "nyuv2_splits.npy"), allow_pickle=True)[()]
    split_indice = {"train": idx["trainNdxs"], "test": idx["testNdxs"]}
    for split in ["test"]:
        index = split_indice[split]
        dic = {}
        object_dir = "%s/%s_object" % (rootdir, split)
        label_dir = "%s/%s_seg%d" % (rootdir, split, labeling)
        object_files = os.listdir(object_dir)
        object_files.sort()
        label_files = os.listdir(label_dir)
        label_files.sort()
        for i in range(len(object_files)):
            label = imread(osj(label_dir, label_files[i]))
            obj = imread(osj(object_dir, object_files[i]))
            obj_label, obj_box = get_object_label_and_box(obj, label)
            dic[int(index[i][0])] = {
                "label": obj_label,
                "bbox": obj_box}
        with open("%s/%s_seg%d_annotations.json" % (rootdir, split, labeling), "w") as fp:
            json.dump(dic, fp)
    """


    """ # Extract object segmentation
    ds = MatNYUv2(rootdir)
    for split in ["train", "test"]:
        ds.split = split
        os.system("rm -r %s/%s_object" % (rootdir, split))
        os.system("mkdir %s/%s_object" % (rootdir, split))
        os.system("rm -r %s/%s_object_viz" % (rootdir, split))
        os.system("mkdir %s/%s_object_viz" % (rootdir, split))
        for i in range(len(ds)):
            _, _, instance, label, index = ds[i]
            instance = np.rot90(instance[0], 3)[:, ::-1]
            label = np.rot90(label[0], 3)[:, ::-1]
            objects = get_instance_mask(label, instance)
            objects_viz = np.zeros(objects.shape[1:])
            for j in range(objects.shape[0]):
                mask = objects[j] > 0
                objects_viz[mask] = j+1
            imwrite(
                "%s/%s_object_viz/%04d.png" % (rootdir, ds.split, index),
                numpy2label(objects_viz, objects.shape[0] + 1))
            imwrite(
                "%s/%s_object/%04d.png" % (rootdir, ds.split, index),
                objects_viz)
    """

    """ # Extract label 40
    ds = NYUV2Dataset("train", rootdir)
    for split in ["train", "test"]:
        ds.split = split
        ds.load_split()
        os.system("rm -r %s/%s_seg40" % (rootdir, split))
        os.system("mkdir %s/%s_seg40" % (rootdir, split))
        os.system("rm -r %s/%s_seg40_viz" % (rootdir, split))
        os.system("mkdir %s/%s_seg40_viz" % (rootdir, split))
        for i in range(len(ds)):
            index = ds.split_indice[ds.split][i][0] - 1
            label = ds.label40_data[index].astype("uint8").copy()
            imwrite("%s/%s_seg40/%04d.png" % (rootdir, ds.split, index + 1), label)
            imwrite("%s/%s_seg40_viz/%04d.png" % (rootdir, ds.split, index + 1), numpy2label(label, 41))
    """

    """ # Extract instance
    ds = MatNYUv2(rootdir)
    for split in ["train", "test"]:
        ds.split = split
        os.system("rm -r %s/%s_instance" % (rootdir, split))
        os.system("mkdir %s/%s_instance" % (rootdir, split))
        os.system("rm -r %s/%s_instance_viz" % (rootdir, split))
        os.system("mkdir %s/%s_instance_viz" % (rootdir, split))
        for i in range(len(ds)):
            index = ds.split_indice[ds.split][i][0] - 1
            instance = ds.instance_data[index][::-1, :].copy()
            imwrite("%s/%s_instance/%04d.png" % (rootdir, ds.split, index + 1),
                np.rot90(instance, 3))
            imwrite("%s/%s_instance_viz/%04d.png" % (rootdir, ds.split, index + 1),
                np.rot90(numpy2label(instance, instance.max()), 3))
    """
    
    """ # Extract depth
    ds = MatNYUv2(rootdir)
    for split in ["train", "test"]:
        ds.split = split
        os.system("rm -r %s/%s_depth_viz" % (rootdir, split))
        os.system("mkdir %s/%s_depth_viz" % (rootdir, split))

        indice = ds.split_indice[ds.split][:, 0] - 1
        depth_data = np.rot90(ds.depth_data[indice, ::-1, :].copy(), 3, axes=(1, 2))
        np.savez("%s/%s_depth.npz" % (rootdir, split), depth_data)
        for i in range(len(ds)):
            index = ds.split_indice[ds.split][i][0] - 1
            depth = depth_data[i]
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = (depth * 255).astype("uint8")
            imwrite("%s/%s_depth_viz/%04d.png" % (rootdir, ds.split, index + 1), depth)
    """