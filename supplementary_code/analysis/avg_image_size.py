import mmcv
from research.utils import build_data
from tqdm import tqdm
import numpy as np

config_ade = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu_secure_aspp.py" # [  3.     512.     673.3855]
config_voc = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_max_pool_secure_aspp.py" #[  3.         512.         713.99861974]
# config_cifar = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet18_cifar100_b_v0/baseline.py" #[32 x 32 x 3]
config_imagenet = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py" # [224 x 224 x 3]
cfg = mmcv.Config.fromfile(config_voc)
dataset = build_data(cfg, mode="test")

a = [list(dataset[sample_id]['img'][0].data.shape) for sample_id in tqdm(range(len(dataset)))]
for x in a:
    if x[1] != 512:
        t = x[1]
        x[1] = x[2]
        x[2] = t

print(np.mean(a, axis=0))