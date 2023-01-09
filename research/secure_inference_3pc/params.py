# class Params:
#     SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet18_8xb32_in1k_finetune.py"
#     MODEL_PATH = "/home/yakir/PycharmProjects/secure_inference/mmlab_models/classification/resnet18_8xb32_in1k_20210831-fbbb1da6.pth"
#     RELU_SPEC_FILE = "/home/yakir/relu_spec_files_storage/classification/resnet18_8xb32_in1k/brelu_0.05.pickle" #"/home/yakir/PycharmProjects/secure_inference/relu_spec_files/classification/block_size_spec_0.15.pickle" #"one
#
#     IMAGE_SHAPE = (1, 3, 224, 224)
#     NUM_IMAGES = 3
#     DUMMY_RELU = False
#     PRF_PREFETCH = False
#     SIMULATED_BANDWIDTH = None #100000000 #None #1000000000 #None #10000000000  # Bits/Second
#

# class Params:
#     # SECURE_CONFIG_PATH = "/home/yakir/Data2/old_work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
#     # MODEL_PATH = "/home/yakir/Data2/old_work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
#     # RELU_SPEC_FILE = None
#     # IMAGE_SHAPE = (1, 3, 256, 256)
#     SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/deeplabv3_m-v2-d8_512x512_160k_ade20k_secure.py"
#
#     # MODEL_PATH = "/home/yakir/iter_128000_finetune.pth"
#     # RELU_SPEC_FILE = None
#
#     MODEL_PATH = "/home/yakir/iter_128000_brelu_0.05.pth"
#     RELU_SPEC_FILE = "/home/yakir/block_size_spec_0.05.pickle"
#     IMAGE_SHAPE = (1, 3, 512, 683)
#
#     NUM_IMAGES = 1
#     DUMMY_RELU = False
#     PRF_PREFETCH = False
#     SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
if IS_TORCH_BACKEND:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
class Params:
    # SECURE_CONFIG_PATH = "/home/yakir/Data2/old_work_dirs/m-v2_256x256_ade20k/baseline/baseline_secure.py"
    # MODEL_PATH = "/home/yakir/Data2/old_work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
    # RELU_SPEC_FILE = None
    # IMAGE_SHAPE = (1, 3, 256, 256)
    SECURE_CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_8xb32_in1k.py"

    # MODEL_PATH = "/home/yakir/iter_128000_finetune.pth"
    # RELU_SPEC_FILE = None

    # MODEL_PATH = "/home/yakir/epoch_14_finetune.pth"
    # RELU_SPEC_FILE = None
    MODEL_PATH = "/home/yakir/epoch_14_brelu.pth"#"/home/yakir/PycharmProjects/secure_inference/mmlab_models/classification/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
    RELU_SPEC_FILE = "/home/yakir/brelu_0.15.pickle"

    IMAGE_SHAPE = (1, 3, 224, 224)

    NUM_IMAGES = 1
    CLIENT_DEVICE = {"cuda":"cuda:0", "cpu":"cpu"}[DEVICE]
    SERVER_DEVICE = {"cuda":"cuda:1", "cpu":"cpu"}[DEVICE]
    CRYPTO_PROVIDER_DEVICE = {"cuda":"cuda:0", "cpu":"cpu"}[DEVICE]
    DUMMY_RELU = False
    DUMMY_MAX_POOL = False
    PRF_PREFETCH = True
    SIMULATED_BANDWIDTH = None #1000000000 #None #10000000000  # Bits/Second
