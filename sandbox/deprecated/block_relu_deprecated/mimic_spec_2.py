import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from research.block_relu.params import ResNetParams
from tqdm import tqdm

params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py",
                      CHECKPOINT="/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth")
import pickle
import numpy as np
content = pickle.load(open("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/reduction_specs/layer_reduction_0.08.pickle", 'rb'))
content_new = dict()

for layer_index, layer_name in enumerate(content.keys()):
    if layer_index == 51:
        print('j')
    r = np.load(f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/channel_knapsack_resblocks/{layer_name}_reduction_to_block_sizes.npy")
    ratio = sum([1/x[0]/x[1] for x in content[layer_name]]) / len(content[layer_name])

    if ratio == 0.000244140625:
        new_b = content[layer_name]
    else:
        new_b = r[-1]
    content_new[layer_name] = new_b

tot = 0
for layer_index, layer_name in enumerate(content_new.keys()):
    ratio = sum([1/x[0]/x[1] for x in content_new[layer_name]]) / len(content_new[layer_name])
    tot+=ratio* params.LAYER_NAME_TO_RELU_COUNT[layer_name]
print(tot/sum(params.LAYER_NAME_TO_RELU_COUNT.values()))
pickle.dump(obj=content_new, file=open("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/reduction_specs/layer_reduction_0.08_channl_group_knapsack.pickle",'wb'))


