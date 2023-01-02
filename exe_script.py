import os
import time

os.chdir("/storage/yakir/secure_inference")
INSTALL_MMDET = "pip install mmdet "
PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/storage/yakir/secure_inference\"; '

# Training - dist - segmentation
# os.system(PYT/esearch/pipeline/dist_train.sh ./research/configs/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/relu_0.05.py 2')

# Training - dist - classification
os.system(PYTHON_PATH_EXPORT +
          'bash ./research/mmlab_tools/classification/dist_train_cls.sh research/configs/classification/resnet/resnet18_8xb32_in1k_brelu_0.05.py 2 ')

# Training - local - classification
# os.system(PYTHON_PATH_EXPORT +
#           'python ./research/mmlab_tools/train_cls.py research/configs/classification/resnet18_8xb16_cifar10/relu_0.15.py')

# Distortion extraction
# os.system(PYTHON_PATH_EXPORT +
#           'python research/distortion/channel_order_distortion.py '
#           f'--batch_index 0 '
#           f'--gpu_id 0 '
#           '--config research/configs/classification/resnet/resnet18_8xb32_in1k.py '
#           '--checkpoint mmlab_models/classification/resnet18_8xb32_in1k_20210831-fbbb1da6.pth '
#           '--output_path outputs/distortions/classification/resnet18_8xb32_in1k/ '
#           '--batch_size 64')

# KnapSack
# os.system(PYTHON_PATH_EXPORT +
#           'python research/knapsack/multiple_choice_knapsack_v2.py '
#           f'--block_size_spec_file_name relu_spec_files/classification/block_size_spec_0.05.pickle '
#           f'--channel_distortion_path outputs/distortions/segmentation/baseline/deeplabv3_m-v2-d8_512x512_160k_ade20k '
#           '--config research/configs/segmentation/baseline/deeplabv3_m-v2-d8_512x512_160k_ade20k.py '
#           '--ratio 0.05 '
#           '--division 512 ')


