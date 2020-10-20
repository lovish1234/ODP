import argparse

import torch
import os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract and save weights")
    folder = '.'
    parser.add_argument(
	"--input",
        default="expr/nyuv2_baseline_26_maskrcnn/model_final.pth",
        help="The path to the checkpoint for test saved during pre-training",
    )
    parser.add_argument(
        "--output",
        default="pretrained/rgb_baseline_26_imagenet.pth",
        help="Output directory for storing model weights",
    )

    args = parser.parse_args()

    checkpoint_path =  args.input
    weight_path = args.output
    
    print("%s => %s" % (checkpoint_path, weight_path))
    checkpoint = torch.load(checkpoint_path)
    torch.save(checkpoint['model'], weight_path) 
