import os.path

from research.secure_inference_3pc.const import IS_TORCH_BACKEND
if IS_TORCH_BACKEND:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class Params:

    SECURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/classification/resnet/resnet50_8xb32_in1k.py")
    MODEL_PATH = None #"/home/yakir/epoch_50.pth"
    RELU_SPEC_FILE = "/home/yakir/4x4.pickle"
    PUBLIC_IP_SERVER = ""
    PUBLIC_IP_CLIENT = ""
    PUBLIC_IP_CRYPTO_PROVIDER = ""
    AWS_DUMMY = False
    IMAGE_SHAPE = (1, 3, 224, 224)
    NUM_IMAGES = 10
    CLIENT_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    SERVER_DEVICE = {"cuda": "cuda:1", "cpu": "cpu"}[DEVICE]
    CRYPTO_PROVIDER_DEVICE = {"cuda": "cuda:0", "cpu": "cpu"}[DEVICE]
    DUMMY_RELU = True
    DUMMY_MAX_POOL = True
    PRF_PREFETCH = True
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
