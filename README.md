# Optional Depth Module for MaskRCNN

Authors: Jianjin Xu, Lovish Chum, Zhaoyang Wang

This is the code base for reproducing the experiment.

This is [demo video](https://youtu.be/4ItCc6NF8Sk).

## Runnning the code

### Preparation

1. Install maskrcnn-benchmark according to the [instructions](https://github.com/AtlantixJJ/OptionalDepthPathway/blob/master/maskrcnn-benchmark/README.md)

2. Prepare the NYUv2 dataset. Under the project root dir execute:

```bash
export DATA_HOME=./maskrcnn-benchmark/datasets/NYUv2/
mkdir $DATA_HOME
cp COCOLikeNYUV2.zip $DATA_HOME
cd $DATA_HOME
unzip COCOLikeNYUV2.zip
wget http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
wget http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
tar -xvf nyu_train_rgb.tgz
tar -xvf nyu_test_rgb.tgz
```

Under `$DATA_HOME` download the depth file from [google drive](https://drive.google.com/open?id=1RZjY7tTGYWqLjkE7_CLqBD6LRkOCCL6j).

3. To train ED-MaskRCNN, you need to download all the estimated depth data from [google drive](https://drive.google.com/open?id=16vwFrIj81yGlH5YvKoXv6YErHAzf10k6) and also place under `$DATA_HOME`.

### Train networks

1. RGB-MaskRCNN

``python tools/train_net.py --config-file "configs/NYUBaselineFT.yaml"``

2. RGBD-MaskRCNN

``python tools/train_net.py --config-file "configs/NYUDepthFT.yaml"``


3. ZD-MaskRCNN

``python tools/train_net.py --config-file "configs/NYUProbDepthFT.yaml``

4. ED-MaskRCNN


Make sure preparation step 3 is completed, then run

```bash
python tools/train_net.py --config-file "configs/NYUProbDenseDepthPretrained.yaml"
```

5. SPADE-MaskRCNN

First make sure RGB-MaskRCNN is trained. If you follow the standard instruction, the RGB-MaskRCNN should be stored in `expr/nyuv2_baseline_26_maskrcnn`. Then run the following command:

```bash
# extract the weight from checkpoint file
python tools/extract_weights.py expr/nyuv2_baseline_26_maskrcnn/model_final.pth pretrained/rgb_baseline_26_imagenet.pth
# option III
python tools/train_net.py --config-file "configs/NYUSPADEFinetune_onrgb.yaml"
# extract the weight from checkpoint file
python tools/extract_weights.py expr/nyuv2_spade_ft_1_imagenet/model_final.pth pretrained/spade_ft1_26_imagenet.pth
# option IV
python tools/train_net.py --config-file "configs/NYUSPADEFinetune_onspade.yaml"
```

### Test the networks

Enter `maskrcnn_benchmark` folder. The script will test all the `.pth` file under `<path of expr dir>` with depth and without depth.

```bash
python tools/testall.py <path of expr> <path of config>
python collect_result.py
```

1. RGB-MaskRCNN

```bash
python tools/testall.py expr/nyuv2_baseline_26_maskrcnn configs/NYUBaselineFT.yaml
```

2. RGBD-MaskRCNN

3. ZD-MaskRCNN

4. ED-MaskRCNN

5. SPADE-MaskRCNN

```bash
# testing model train by option I+II+III
python tools/testall.py expr/nyuv2_spade_ft_1_imagenet configs/NYUSPADE_26.yaml
# testing model trained by option I+II+III+IV
python tools/testall.py expr/nyuv2_spade_ft_2_imagenet configs/NYUSPADE_26.yaml
```

### Visualize the results



This script will visualize all the results collected by `collect_result.py`.

```bash
python tools/visualize.py
```
