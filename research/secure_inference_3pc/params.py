class Params:
    CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline.py"
    SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    MODEL_PATH = "/home/yakir/iter_80000.pth"
    RELU_SPEC_FILE = None #"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/test/block_size_spec_0.15.pickle"
    IMAGE_SHAPE = (1, 3, 256, 256)
    NUM_IMAGES = 4
    DUMMY_RELU = False
    PRF_PREFETCH = False
    SIMULATED_BANDWIDTH = 100000000  # Bits/Second
