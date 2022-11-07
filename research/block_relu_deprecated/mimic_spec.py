import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import numpy as np
content = pickle.load(open("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/reduction_specs/layer_reduction_0.08.pickle", 'rb'))
content_new = dict()

for layer_index, layer_name in enumerate(content.keys()):
    if layer_index == 51:
        print('j')
    r = np.load(f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/channel_knapsack_resblocks/{layer_name}_reduction_to_block_sizes.npy")
    ratio = sum([1/x[0]/x[1] for x in content[layer_name]]) / len(content[layer_name])
    print(ratio)
    if ratio == 0.000244140625:
        new_b = content[layer_name]
    else:
        new_b = r[int(ratio * 1000)]
    content_new[layer_name] = new_b

pickle.dump(obj=content_new, file=open("/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/V1/reduction_specs/layer_reduction_0.08_channl_group_knapsack.pickle",'wb'))


