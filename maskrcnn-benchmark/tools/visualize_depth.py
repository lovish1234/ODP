import numpy as np
from PIL import Image
from matplotlib import cm
import os

if __name__=='__main__':


	x = np.load('/proj/vondrick/lovish/datasets/NYUv2/test_depth.npz', allow_pickle=True)['arr_0']
	
	for i in range(0,100):	

		normalized_image = (x[i]-np.min(x[i]))/(np.max(x[i])-np.min(x[i]))
		im = Image.fromarray(np.uint8(cm.rainbow(normalized_image)*255))

		folder = 'depth'
		name = 'depth_image_'+str(i)+'.png'

		im.save(os.path.join(folder, name))