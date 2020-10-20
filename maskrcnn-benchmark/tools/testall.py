import os
import sys
from os.path import join as osj 

config_files = [sys.argv[2]]
ckpt_dirs = [sys.argv[1]]
cmds = []
for ckpt_dir in ckpt_dirs:
    for use_depth in [0, 1]:
        for is_train in [0]:
            config_file = config_files[is_train]
            depth = "depth" if use_depth == 1 else "no_depth"
            train = "train" if is_train == 1 else "test"
            models = [m for m in os.listdir(ckpt_dir) if ".pth" in m]
            for model in models:
                cmd = f"python tools/test_net.py --use_depth {use_depth} --config-file {config_file} --ckpt {ckpt_dir}/{model} OUTPUT_DIR {ckpt_dir}/{depth}/{train}/{model} TEST.IMS_PER_BATCH 1"

                cmds.append(cmd)

for i,cmd in enumerate(cmds):
    print(cmd)
    os.system(cmd)