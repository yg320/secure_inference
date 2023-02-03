import os.path

from research.secure_inference_3pc.const import IS_TORCH_BACKEND
if IS_TORCH_BACKEND:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class Params:
    # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py")
    # MODEL_PATH = None #"/home/yakir/PycharmProjects/secure_inference/mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth" #None #"/home/yakir/epoch_50.pth"
    # RELU_SPEC_FILE = "/home/ubuntu/specs/seg_specs/6x6.pickle"

    # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_8xb32_in1k.py")
    # MODEL_PATH = None #"/home/yakir/epoch_50.pth"
    # RELU_SPEC_FILE = "/home/yakir/4x4.pickle"

    SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_8xb32_in1k.py")
    MODEL_PATH = None #"/home/yakir/epoch_14_avg_pool.pth"
    RELU_SPEC_FILE = "/home/ubuntu/specs/cls_specs/2x3.pickle"

    # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_8xb32_in1k_maxpool.py")
    # MODEL_PATH = None #"/home/yakir/PycharmProjects/secure_inference/mmlab_models/classification/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
    # RELU_SPEC_FILE = None

    PUBLIC_IP_SERVER = ""
    PUBLIC_IP_CLIENT = ""
    PUBLIC_IP_CRYPTO_PROVIDER = ""
    AWS_DUMMY = True
    NUM_IMAGES = 3
    CLIENT_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    SERVER_DEVICE = {"cuda": "cuda:1", "cpu": "cpu"}[DEVICE]
    CRYPTO_PROVIDER_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    DUMMY_RELU = False
    PRF_PREFETCH = True
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
