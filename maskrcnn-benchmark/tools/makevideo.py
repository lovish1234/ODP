"""
Combine 4 video into 1
"""
import numpy as np
import glob
import argparse
from os.path import join as osj
from moviepy.editor import VideoClip
from PIL import Image
import cv2 as cv

def imread(fpath):
    with open(osj(fpath), "rb") as f:
        return np.asarray(Image.open(f).resize((640, 480)).convert("RGB"))

FONT_HEIGHT = 40
FPS = 30
TOTAL_TIME = 500 // FPS
PAD_SIZE = 10
IMSIZE = (480, 640)
CANVAS_SIZE = ((IMSIZE[0] + PAD_SIZE + FONT_HEIGHT) * 2, IMSIZE[1] * 2 + PAD_SIZE)

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="demo.mp4")
args = parser.parse_args()

root = "datasets/ODPDemo"
video_dirs = ["rgb_baseline", "depth", "spade_no_depth", "spade_depth"]
title = ["RGB Baseline", "Depth", "SPADE w/o depth", "SPADE w/ depth"]

video_files = []
for vd in video_dirs:
    files = glob.glob(osj(root, vd, "*.png"))
    files.sort()
    video_files.append(files)

def make_frame(t):
    index = int(FPS * t)
    canvas = np.zeros(CANVAS_SIZE + (3,), dtype="uint8")
    canvas.fill(255)

    for i in range(len(video_dirs)):
        im = imread(video_files[i][index])
        
        x, y = i // 2, i % 2 # hardcoded for 4 videos
        p = [x * (IMSIZE[0] + PAD_SIZE + FONT_HEIGHT), y * (IMSIZE[1] + PAD_SIZE)]

        p[0] += FONT_HEIGHT
        p[1] += IMSIZE[1] // 2 - 100
        canvas = cv.putText(canvas, title[i], (p[1], p[0]),
                cv.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2, cv.LINE_AA)
        p[1] -= IMSIZE[1] // 2 - 100
        p[0] += PAD_SIZE
        
        #stx, sty = x * (IMSIZE[0] + PAD_SIZE), y * (IMSIZE[1] + PAD_SIZE)
        canvas[p[0]:p[0]+IMSIZE[0], p[1]:p[1]+IMSIZE[1]] = im



    return canvas

animation = VideoClip(make_frame, duration=TOTAL_TIME)
animation.write_videofile(args.output, fps=FPS)