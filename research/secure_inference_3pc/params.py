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
    # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py")
    # MODEL_PATH = "/home/yakir/epoch_50.pth"
    # RELU_SPEC_FILE = "/home/yakir/4x4.pickle"

    # # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_8xb32_in1k.py")
    # # MODEL_PATH = "/home/yakir/epoch_50.pth"
    # # RELU_SPEC_FILE = "/home/yakir/4x4.pickle"
    # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py")
    # # MODEL_PATH = "/home/yakir/epoch_50.pth"
    # # RELU_SPEC_FILE = "/home/yakir/4x4.pickle"
    # MODEL_PATH = "/home/yakir/epoch_14_avg_pool.pth"
    # RELU_SPEC_FILE = None #"/home/ubuntu/specs/cls_specs/2x3.pickle"
    #
    # # SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py")
    # # MODEL_PATH ="/home/yakir/PycharmProjects/secure_inference/mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth"
    # # RELU_SPEC_FILE = None

    # SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu_secure_aspp.py"
    # MODEL_PATH = "/home/yakir/assets/mobilenet_ade/models/ratio_0.09.pth"
    # RELU_SPEC_FILE = '/home/yakir/assets/mobilenet_ade/block_spec/0.09.pickle'
    # NUM_IMAGES = 2000

    SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py"
    MODEL_PATH = "/home/yakir/assets/resnet_imagenet/models/ratio_0.12.pth"
    RELU_SPEC_FILE = '/home/yakir/assets/resnet_imagenet/block_spec/0.12.pickle'
    NUM_IMAGES = 50


    AWS_DUMMY = False
    CLIENT_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    SERVER_DEVICE = {"cuda": "cuda:1", "cpu": "cpu"}[DEVICE]
    CRYPTO_PROVIDER_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    DUMMY_RELU = False
    PRF_PREFETCH = False
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
