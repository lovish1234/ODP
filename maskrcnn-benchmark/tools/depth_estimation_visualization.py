import numpy as np
import glob 
from os.path import join as osj
import os 
from PIL import Image

test_depth_estimation = np.load('/Users/ouchouyang/Downloads/NYUv2/test_depth_estimation_megadepth.npz')
test_depth_estimation = test_depth_estimation['arr_0']



num_images, height, width = test_depth_estimation.shape


#test_img_names = sorted(glob.glob("/Users/ouchouyang/Downloads/NYUv2/nyu_test_rgb/*.png"))

test_img_names = [os.path.basename(x) for x in sorted(glob.glob("/Users/ouchouyang/Downloads/NYUv2/nyu_test_rgb/*.png"))]

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

for i in range(num_images):
    depth = test_depth_estimation[i,:,:]
    depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
    depth = (depth * 255).astype("uint8")
    imwrite("/Users/ouchouyang/Downloads/NYUv2/depth_estimation_visualization/%s" % (test_img_names[i]),depth)

    #imwrite("%s/%s_depth_viz/%04d.png" % (rootdir, ds.split, index + 1), depth)
