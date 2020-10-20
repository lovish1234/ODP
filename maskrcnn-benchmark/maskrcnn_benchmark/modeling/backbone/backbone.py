# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import spade_resnet
#from . import experiment3
#from . import depth_estimator
#from . import pytorch_DIW_scratch
# [Jianjin Xu] This cause error because TrainOptions conflict with global argparse, please fix
#from . import experiment3
from . import baseline_resnet
from . import probabilistic_depth_resnet
from . import depth_resnet
from . import depth_resnet_pretrained
from . import probabilistic_depth_resnet_pretrained
from . import probabilistic_depth_estimation_resnet



@registry.BACKBONES.register("RPDE-50-FPN")
def build_prob_depth_estimation_resnet_backbone(cfg):
    body = depth_resnet.resnet50(pretrained=False) # default depth prob = 0.5
    body.depth_prob = cfg.MODEL.BACKBONE.DEPTH_PROB
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("EXP3-50-FPN")
def build_experiment3_backbone(cfg):

    body = experiment3.resnet50(pretrained=True) ## load model with pretrained DE and pretrained reset50

    # add one more convolution channel for depth but preserving weights for RGB
    weights_rgb = body.conv1.weight.data
    depth_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    depth_conv1.weight.requires_grad=False
    depth_conv1.weight[:,:3] = weights_rgb
    depth_conv1.weight.requires_grad=True
    body.conv1 = depth_conv1


    ## after loading weights, change the model by adding a depth channel

    ## use first three channels to produce the 4th channel and pass that channel (evaluation mode) to expanded conv1 kernel
    ## or use training mode to let gradient flow to depth estimator to train


    #model = pytorch_DIW_scratch.pytorch_DIW_scratch
    #model_parameters = self.load_network()
    #model.load_state_dict(model_parameters)

    # takes RGB and output RGBD 
    #DE = depth_estimator.megadepth(pretrained = True)

    #body = nn.Sequential ([DE,body])

    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )

    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels

    return model

@registry.BACKBONES.register("SPADER-50-FPN")
def build_spade_resnet_backbone(cfg):
    body = spade_resnet.resnet50(pretrained=True)
    body.depth_prob = cfg.MODEL.BACKBONE.DEPTH_PROB
    body.fix_iter = cfg.MODEL.BACKBONE.FIX_ITER

    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )

    # append resent with the feature pyramid network
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("SPADER-DHHA-50-FPN")
def build_spade_hha_resnet_backbone(cfg):
    body = spade_resnet.resnet50(pretrained=True, dim=4)
    body.depth_prob = cfg.MODEL.BACKBONE.DEPTH_PROB
    
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )

    # append resent with the feature pyramid network
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("RPD-50-FPN")
def build_prob_depth_resnet_backbone(cfg):

    # pass on the 
    # depth_prob = cfg.BACKBONE.DEPTH_PROB
    # rng = np.random.RandomState(1314)
    # no_depth = (self.rng.rand() < self.depth_prob)

    # get the depth probability from config
    depth_prob = cfg.MODEL.BACKBONE.DEPTH_PROB

    # pass the depth probability here

    if cfg.MODEL.RESNETS.PRETRAINED:

        body = probabilistic_depth_resnet_pretrained.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED, depth_prob = depth_prob)
        #body = depth_resnet.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED)
        weights_rgb = body.conv1.weight.data

        depth_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        depth_conv1.weight.requires_grad=False

        depth_conv1.weight[:,:3] = weights_rgb
        depth_conv1.weight.requires_grad=True
        body.conv1 = depth_conv1
    else:
        body = probabilistic_depth_resnet.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED, depth_prob = depth_prob)



    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("RD-50-FPN")
def build_depth_resnet_backbone(cfg):

    if cfg.MODEL.RESNETS.PRETRAINED:

        body = depth_resnet_pretrained.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED)
        #body = depth_resnet.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED)
        weights_rgb = body.conv1.weight.data

        depth_conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        depth_conv1.weight.requires_grad=False

        depth_conv1.weight[:,:3] = weights_rgb
        depth_conv1.weight.requires_grad=True
        body.conv1 = depth_conv1
    else:
        body = depth_resnet.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED)


    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("RB-50-FPN")
def build_baseline_resnet_backbone(cfg):
    body = baseline_resnet.resnet50(pretrained=cfg.MODEL.RESNETS.PRETRAINED)

    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )

    # append resent with the feature pyramid network
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model



@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
