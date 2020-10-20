import torch
rgb = torch.load("pretrained/rgb_baseline.pth", map_location="cpu")
spade = torch.load("expr/nyuv2_spade_maskrcnn_ft/model_0000010.pth", map_location="cpu")
spade = spade['model']
for k in rgb.keys():
    if not (rgb[k] == spade[k]).all():
        print(k)
        print((rgb[k] - spade[k]).abs().max())