import os

os.system("rm -r results")

result_dir = "results/spade"
target_dirs = ["nyuv2_spade_ft_1_imagenet", "nyuv2_spade_ft_2_imagenet"] #["0.1_0.9", "0.5_0.5", "0.9_0.1"]
rename_dirs = ["nyuv2_spade_ft_1_imagenet", "nyuv2_spade_ft_2_imagenet"] #["_0.1", "", "_0.9"]
target_subdirs = ["test"]
frdir = "expr/%s/%s/%s/%s/inference/"
todir = f"{result_dir}/%s/%s/%s/%s"

for i,td in enumerate(target_dirs):
    for j, ts in enumerate(target_subdirs):
        for depth in ["depth", "no_depth"]:
            models = os.listdir(f"expr/{rename_dirs[i]}/{depth}/{ts}")
            models.sort()
            for model in models:
                print(f"mkdir -p {result_dir}/{td}/{depth}/{ts}/{model}")
                os.system(f"mkdir -p {result_dir}/{td}/{depth}/{ts}/{model}")
                frdir_ = frdir % (rename_dirs[i], depth, ts, model)
                frdir_ = frdir_ + os.listdir(frdir_)[0]
                todir_ = todir % (td, depth, ts, model)
                print(f"cp {frdir_}/* {todir_}/")
                os.system(f"cp {frdir_}/* {todir_}")
