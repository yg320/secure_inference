import matplotlib
matplotlib.use("TkAgg")

from research.block_relu.params import ResNetParams
from research.pipeline.backbones.secure_resnet import MyResNet  # TODO: find better way to init


gpu_id = 0
checkpoint_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/deeplabv3_r50-d8_512x512_4x4_80k_coco-stuff164k/iter_80000.pth"
config = "/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/my_resnet_coco-stuff_164k.py"
params = ResNetParams(HIERARCHY_NAME=None,
                      LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS=None,
                      LAYER_HIERARCHY_SPEC=None,
                      DATASET="coco_stuff164k",
                      CONFIG=config,
                      CHECKPOINT=checkpoint_path)


layer_name = 'layer1_0_1'
next_layer_name = 'layer1_0_2'

m = params.LAYER_NAME_TO_LAYER_DIM[layer_name]
i = params.LAYER_NAME_TO_CHANNELS[layer_name]
o = params.LAYER_NAME_TO_CHANNELS[next_layer_name]
f = 3
q = m - f + 1
comm_costs = []
for i in range(params.LAYER_NAME_TO_CHANNELS[next_layer_name] + 1):
    comm_cost = 2 * m ** 2 * i + 2 * f **2 * o * i + q ** 2 * o
    comm_costs.append(comm_cost)

plt.plot(comm_costs)