import collections
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import  pyplot as plt

import numpy as np
from research.block_relu.params import MobileNetV2Params
import numpy as np
from tqdm import tqdm
params = MobileNetV2Params()

CHANNEL_TO_LAYER = []
CHANNEL_ORD_TO_CHANNEL = []
for layer in params.LAYER_NAMES:
    CHANNEL_TO_LAYER.extend([layer] * params.LAYER_NAME_TO_CHANNELS[layer])
    CHANNEL_ORD_TO_CHANNEL.extend(np.arange(params.LAYER_NAME_TO_CHANNELS[layer]))

index_to_block_sizes  = np.load('/home/yakir/Data2/DP/index_to_block_sizes.npy')
W  =np.load(f"/home/yakir/Data2/DP/W.npy")

NUM_CHANNELS = 4288
relus_count = 4000000 - 1
channel_to_block_size = np.zeros((NUM_CHANNELS, 2))
channel_to_index = np.zeros(NUM_CHANNELS)

for block in tqdm([4287, 4200, 4100, 4000, 3900, 3800, 3700, 3600, 3500, 3400, 3300, 3200, 3100, 3000, 2900, 2800, 2700, 2600, 2500, 2400, 2300, 2200, 2100, 2000, 1900, 1800, 1700, 1600, 1500, 1400, 1300, 1200, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]):
    dp_ = np.load(f"/home/yakir/Data2/DP/dp_{block}.npy")
    dp_arg_ = np.load(f"/home/yakir/Data2/DP/dp_arg_{block}.npy")

    if block == 4287:
        channels_to_use = 88
        reduce = 87
    else:
        channels_to_use = 100
        reduce = 100
    for channel in reversed(range(channels_to_use)):

        channel_real = (block - reduce) + channel
        arg = dp_arg_[channel, relus_count]

        channel_num_relus = W[channel_real, arg]
        relus_count -= channel_num_relus
        channel_to_block_size[channel_real] = index_to_block_sizes[arg, channel_real]
        channel_to_index[channel_real] = arg

block_spec = dict()
for channel_ord in range(NUM_CHANNELS):
    layer = CHANNEL_TO_LAYER[channel_ord]
    channel = CHANNEL_ORD_TO_CHANNEL[channel_ord]
    if layer not in block_spec.keys():
        block_spec[layer] = np.zeros(shape=(params.LAYER_NAME_TO_CHANNELS[layer], 2), dtype=np.int32)
    block_spec[layer][channel] = channel_to_block_size[channel_ord]

import pickle
pickle.dump(obj=block_spec, file=open("/home/yakir/Data2/DP/block_spec.pickle", 'wb'))
# np.save(file="/home/yakir/Data2/DP/block_spec.pickle", arr=block_spec)
derived_relu_count = 0
relus = collections.defaultdict(list)
for channel in tqdm(range(NUM_CHANNELS)):
    # derived_relu_count += (W[channel, channel_to_index[channel]])
    layer_name = CHANNEL_TO_LAYER[channel]
    relus[layer_name].append(W[channel, channel_to_index[channel].astype(np.int32)])

x = []
for k, v in relus.items():
    x.append(sum(v)/params.LAYER_NAME_TO_RELU_COUNT[k])

plt.bar(range(len(x[:-1])), x[:-1])