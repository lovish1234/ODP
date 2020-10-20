"""
Image depth dataset (no annotation)
"""
import torch
import torchvision
import torchvision.utils as vutils
from os.path import join as osj
import glob
import numpy as np
from PIL import Image

def imread(fpath):
    with open(osj(fpath), "rb") as f:
        return np.asarray(Image.open(f))
        
class ImageDepthDataset(object):
    def __init__(self, root, image_dir="image", depth_file="depth.npy", **kwargs):
        self.root = root
        self.image_dir = osj(root, image_dir)
        self.image_files = glob.glob(osj(self.image_dir, "*.png"))
        self.image_files.sort()
        self.depth_file = osj(root, depth_file)
        self.depth_data = np.load(self.depth_file)
        minimum, maximum = self.depth_data.min(), self.depth_data.max()
        self.depth_data = (self.depth_data - minimum) / (maximum - minimum) * 2 - 1
    
    def __len__(self):
        return self.depth_data.shape[0]

    def __getitem__(self, idx):
        image = imread(self.image_files[idx])
        depth = self.depth_data[idx]
        image = (torch.from_numpy(image).float() - 127.5) / 127.5
        image = image.permute(2, 0, 1)
        depth = torch.from_numpy(depth).float().unsqueeze(0)

        image_ = torch.cat([image, depth])

        return image_, 0, idx
    
    def get_img_info(self, i):
        return {"height": 480, "width": 640}