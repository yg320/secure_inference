import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from research.block_relu.distortion_utils import DistortionUtils
from research.block_relu.params import ResNetParams
from research.pipeline.backbones.secure_resnet import MyResNet  # TODO: find better way to init
from tqdm import tqdm
import pickle

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

gpu_id = 1
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

layer_index_to_block_spec_index = np.array([999] * len(params.LAYER_NAMES))
block_size_spec = {layer_name:layer_name_to_block_sizes[layer_name][layer_index_to_block_spec_index[layer_index]] for layer_index, layer_name in enumerate(params.LAYER_NAMES)}

all_relus = []
all_losses = []
all_costs = []
all_layer_index_to_block_spec_index = []
batches_ind = 118280//8
counter = 0
lambd = 10
first_point = 20
np.random.seed(123)
for _ in tqdm(range(10000000)):
    layer_index = np.random.randint(0, 57)
    layer_name = params.LAYER_NAMES[layer_index]
    sample_points = np.hstack([[layer_index_to_block_spec_index[layer_index]], [first_point], np.random.randint(first_point+1, 999, 18), [999]])
    losses = []
    relus = []
    batch = np.random.choice(batches_ind, size=4, replace=False)
    try:
        for x in sample_points:
            block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][x]
            losses.append(np.array(
                [distortion_utils.get_batch_distortion(block_size_spec, batch_index=batch_index)['losses_distorted'] for
                 batch_index in batch]).mean())
            relus.append(get_spec_ratio(block_size_spec, params))
    except ValueError:
        continue
    losses = np.array(losses)
    relus = np.array(relus)
    cost = losses + lambd * relus

    cost_arg_min = np.argmin(cost)
    index = sample_points[cost_arg_min]
    all_relus.append(relus[cost_arg_min])
    all_losses.append(losses[cost_arg_min])
    all_costs.append(cost[cost_arg_min])
    layer_index_to_block_spec_index[layer_index] = index
    all_layer_index_to_block_spec_index.append([x for x in layer_index_to_block_spec_index])
    block_size_spec[layer_name] = layer_name_to_block_sizes[layer_name][index]

    pickle.dump(file=open(f"/home/yakir/tmp/dumping_stuff_{lambd}_{4}.pickle", 'wb'), obj=(
        all_relus,
        all_losses,
        all_costs,
        all_layer_index_to_block_spec_index,
        block_size_spec
    ))

    print(counter, all_relus[-1], all_losses[-1], all_costs[-1])
    counter += 1
