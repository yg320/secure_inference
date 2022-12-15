import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from research.block_relu.distortion_utils import DistortionUtils
from research.block_relu.params import ResNetParams
from research.pipeline.backbones.secure_resnet import MyResNet  # TODO: find better way to init
from tqdm import tqdm

def get_spec_ratio(block_size_spec, params):
    # TODO: handle boundaries
    tot_relus = sum(params.LAYER_NAME_TO_RELU_COUNT.values())
    relus = 0
    for layer_name in params.LAYER_NAMES:
        ratio = (sum([1/x[0]/x[1] for x in block_size_spec[layer_name]]) / len(block_size_spec[layer_name]))
        relus += params.LAYER_NAME_TO_RELU_COUNT[layer_name] * ratio
    return relus / tot_relus

def get_cost(block_size_spec, lambd, batch_count):
    loss = np.array(
        [distortion_utils.get_batch_distortion(block_size_spec, batch_index=batch_index)['losses_distorted'] for batch_index in
         range(batch_count)]).mean()

    relus = get_spec_ratio(block_size_spec, params)
    return loss - lambd * relus

gpu_id = 0
checkpoint_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth"
config = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py"
params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG=config,
                      CHECKPOINT=checkpoint_path)

distortion_utils = DistortionUtils(gpu_id, params)

layer_name_to_block_sizes = dict()
for layer_name in params.LAYER_NAMES:
    layer_name_to_block_sizes[layer_name] = np.load(f"/home/yakir/Data2/assets_v3/deformations/coco_stuff164k/ResNetV1c/channel_knapsack/{layer_name}_reduction_to_block_sizes.npy")[1:]

block_size_spec = {layer_name:layer_name_to_block_sizes[layer_name][100] for layer_name in params.LAYER_NAMES}

sample_points = [10, 100, 400]

coef_mat = np.array([[(sample_points[0]/1000)**2, (sample_points[0]/1000), 1],
                     [(sample_points[1]/1000)**2, (sample_points[1]/1000), 1],
                     [(sample_points[2]/1000)**2, (sample_points[2]/1000), 1]])
inv = np.linalg.inv(coef_mat)
roundd = 0
rrr = []
while True:
    roundd += 1
    rrr.append({k:v.copy() for k,v in block_size_spec.items()})
    l = np.array(
        [distortion_utils.get_batch_distortion(block_size_spec, batch_index=batch_index)['losses_distorted'] for
         batch_index in
         range(16)]).mean()
    print(f'Round - {roundd}', get_spec_ratio(block_size_spec, params), l)




    for layer_name in tqdm(params.LAYER_NAMES):
        losses = []
        relus = []
        for x in sample_points:
            block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][x]
            losses.append(np.array(
                [distortion_utils.get_batch_distortion(block_size_spec, batch_index=batch_index)['losses_distorted'] for batch_index in
                 range(16)]).mean())
            relus.append(get_spec_ratio(block_size_spec, params))

        losses = np.array(losses)
        relus = np.array(relus)
        cost = losses + 25 * relus

        if cost[2] <= min(cost[0], cost[1]):
            index = sample_points[2]
        elif cost[0] <= min(cost[1], cost[2]):
            index = sample_points[0]
        else:
            a, b, c = inv @ cost
            index = int(1000 * (-b / (2*a)))
            # print(index)

        block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][index]


# array([ 0.43160079, -0.26342498,  1.13001939])
# a, b, _ = a @ np.array([1.1274283, 1.1079929, 1.1062071])
# -b / 2*a
# 0.056847214395624984
# 1000 * (-b / 2*a)
# 56.84721439562498
#
# 56
plt.plot([10,100,200, 300, 400, 500, 600, 700, 999], losses)
plt.plot(np.arange(0,1000,100), losses-30*relus)
block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][b]
cost_b = get_cost(block_size_spec, lambd, 1)
block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][c]
cost_c = get_cost(block_size_spec, lambd, 1)

block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][500]
print(get_cost(block_size_spec, lambd))
block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][-1]
print(get_cost(block_size_spec, lambd))
block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][250]
print(get_cost(block_size_spec, lambd))
block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][50]
print(get_cost(block_size_spec, lambd))
loss = np.array([distortion_utils.get_batch_distortion(block_size_spec, batch_index=0)['losses_distorted'] for batch_index in range(4)]).mean()
print(loss)
