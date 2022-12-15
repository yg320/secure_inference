
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import pickle
import numpy as np


layer_groups = [
    ['conv1', 'layer1_0_0', 'layer2_0_0', 'layer2_0_1', 'layer2_1_0', 'layer2_1_1', 'layer3_0_0', 'layer3_0_1',
     'layer3_1_0', 'layer3_1_1', 'layer3_2_0', 'layer3_2_1'],
    ['layer4_0_0', 'layer4_0_1', 'layer4_1_0', 'layer4_1_1', 'layer4_2_0', 'layer4_2_1', 'layer4_3_0', 'layer4_3_1',
     'layer5_0_0', 'layer5_0_1', 'layer5_1_0', 'layer5_1_1', 'layer5_2_0', 'layer5_2_1'],
    ['layer6_0_0', 'layer6_0_1', 'layer6_1_0', 'layer6_1_1', 'layer6_2_0', 'layer6_2_1', 'layer7_0_0', 'layer7_0_1'],
    ['decode_0', 'decode_1', 'decode_2', 'decode_3', 'decode_4', 'decode_5']
]

plt.figure()
for layer_group_index, layer_group in enumerate(layer_groups):

    noises = []
    losses_distorted = []
    for seed in range(18):
        cur_losses_distorted = []
        cur_noises = []

        for batch_index in range(128):

            f = f"/home/yakir/distortion_approximation_v2/final_approximation/{layer_group_index}_{seed}_{batch_index}.pickle"
            content = pickle.load(open(f, 'rb'))

            layer_name_additive_assets = content['layer_name_additive_assets']

            if len(layer_name_additive_assets) > 0:
                key_to_use = list(list(layer_name_additive_assets.values())[0][0]['noises_distorted'].keys())[0]

                tmp = []
                for k, v in layer_name_additive_assets.items():
                    for cur_v in v:
                        tmp.append(cur_v['noises_distorted'][key_to_use])
                tmp = np.array(tmp).sum(axis=0)
                cur_noises.append(tmp)
            cur_losses_distorted.append(np.array(content["assets"]['losses_distorted']))

        cur_noises = np.array(cur_noises).flatten()
        cur_losses_distorted = np.array(cur_losses_distorted).flatten()
        losses_distorted.append(cur_losses_distorted)
        noises.append(cur_noises)
    noises = np.array(noises)
    losses_distorted = np.array(losses_distorted)
    x = str(np.corrcoef(noises, losses_distorted)[0,1])[:5]

    plt.scatter(noises, losses_distorted, alpha=0.3, label=f"layer_group = {layer_group_index} corr={x}")
    plt.legend()
