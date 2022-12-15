from research.block_relu.params import ResNetParams
import pickle
import numpy as np

BLOCKS_LAYERS = \
    [
        ["stem_2",
         "stem_5",
         "stem_8"],
        ["layer1_0_1",
         "layer1_0_2",
         "layer1_0_3"],
        ["layer1_1_1",
         "layer1_1_2",
         "layer1_1_3"],
        ["layer1_2_1",
         "layer1_2_2",
         "layer1_2_3"],
        ["layer2_0_1",
         "layer2_0_2",
         "layer2_0_3"],
        ["layer2_1_1",
         "layer2_1_2",
         "layer2_1_3"],
        ["layer2_2_1",
         "layer2_2_2",
         "layer2_2_3"],
        ["layer2_3_1",
         "layer2_3_2",
         "layer2_3_3"],
        ["layer3_0_1",
         "layer3_0_2",
         "layer3_0_3"],
        ["layer3_1_1",
         "layer3_1_2",
         "layer3_1_3"],
        ["layer3_2_1",
         "layer3_2_2",
         "layer3_2_3"],
        ["layer3_3_1",
         "layer3_3_2",
         "layer3_3_3"],
        ["layer3_4_1",
         "layer3_4_2",
         "layer3_4_3"],
        ["layer3_5_1",
         "layer3_5_2",
         "layer3_5_3"],
        ["layer4_0_1",
         "layer4_0_2",
         "layer4_0_3"],
        ["layer4_1_1",
         "layer4_1_2",
         "layer4_1_3"],
        ["layer4_2_1",
         "layer4_2_2",
         "layer4_2_3"],
        ["decode_0",
         "decode_1",
         "decode_2",
         "decode_3",
         "decode_4",
         "decode_5"]
    ]


params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py",
                      CHECKPOINT="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth")



content = pickle.load(open("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/reduction_specs/layer_reduction_0.08.pickle", 'rb'))

group_stuff = []
for layer_group in range(18):
    layer_names = BLOCKS_LAYERS[layer_group]
    tot_relus = 0
    baseline_relus = 0
    for layer_name in layer_names:
        ratio = sum([1 / x[0] / x[1] for x in content[layer_name]]) / len(content[layer_name])
        num_relus = params.LAYER_NAME_TO_RELU_COUNT[layer_name]
        tot_relus += num_relus * ratio
        baseline_relus += num_relus
    print(tot_relus / baseline_relus)
    group_stuff.append(tot_relus / baseline_relus)