class Params:
    SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet18_8xb16_cifar10/resnet18_8xb16_cifar10.py"
    MODEL_PATH = "/home/yakir/PycharmProjects/secure_inference/mmlab_models/classification/resnet18_b16x8_cifar10_20210528-bd6371c8.pth"
    RELU_SPEC_FILE = None

    IMAGE_SHAPE = (1, 3, 32, 32)
    NUM_IMAGES = 11
    DUMMY_RELU = True
    PRF_PREFETCH = False
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second


# class Params:
#     # CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
#     SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
#     MODEL_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
#     RELU_SPEC_FILE = None
#     IMAGE_SHAPE = (1, 3, 256, 256)
#     NUM_IMAGES = 1
#     DUMMY_RELU = True
#     PRF_PREFETCH = False
#     SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
