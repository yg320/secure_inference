import pickle
import numpy as np

CHANNELS_HALF = True
content = pickle.load(open("/home/yakir/tesnet18/distortions/resnet18_2xb64_cifar100/block_sizes/S1.pickle", "rb"))
for k in content:
    content[k][:, 0] = 0
    content[k][:, 1] = 1

    if CHANNELS_HALF:
        content[k] = content[k][:content[k].shape[0]//2]

layers_to_keep = ['layer2_0_1', 'layer2_1_1', 'layer3_0_1', 'layer3_1_1']
# layers_to_keep = ['layer2_0_2', 'layer2_1_2', 'layer3_0_2', 'layer3_1_2']

for layer in layers_to_keep:
    content[layer] = np.ones_like(content[layer])

pickle.dump(obj=content, file=open("/home/yakir/tiny_tesnet18/distortions/conv_stride_2/0K.pickle", "wb"))