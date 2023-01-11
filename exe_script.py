import os
import time

os.chdir("/storage/yakir/secure_inference")
INSTALL_MMDET = "pip install mmdet "
PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:/storage/yakir/secure_inference\"; '
time.sleep(24*3600*7)
# Training - dist - segmentation
# os.system(PYT/esearch/pipeline/dist_train.sh ./research/configs/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/relu_0.05.py 2')

# Training - dist - classification
# os.system(PYTHON_PATH_EXPORT +
#           'bash ./research/mmlab_tools/classification/dist_train_cls.sh research/configs/classification/resnet/resnet18_8xb32_in1k_brelu_0.05.py 2 ')

# Training - local - classification
# os.system(PYTHON_PATH_EXPORT +
#           'python ./research/mmlab_tools/train_cls.py research/configs/classification/resnet18_8xb16_cifar10/relu_0.15.py')

# Distortion extraction
# os.system(PYTHON_PATH_EXPORT +
#           'python research/distortion/channel_order_distortion.py '
#           '--batch_index 7 '
#           '--gpu_id 0 '
#           '--config research/configs/classification/resnet/resnet50_8xb32_in1k_finetune_0.0001_avg_pool.py '
#           '--checkpoint outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_13.pth '
#           '--output_path outputs/distortions/classification/resnet50_8xb32_in1k_finetune_0.0001_avg_pool/ '
#           '--batch_size 64 '
#           '--cur_iter 1 '
#           '--num_iters 5 ')


# KnapSack
if False:
    os.system(PYTHON_PATH_EXPORT +
              'python research/knapsack/multiple_choice_knapsack_v2.py '
              '--block_size_spec_file_name relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_1/block_size_spec_0.1.pickle '
              '--channel_distortion_path outputs/distortions/classification/resnet50_8xb32_in1k_finetune_0.0001_avg_pool/ '
              '--config research/configs/classification/resnet/resnet50_8xb32_in1k_finetune_0.0001_avg_pool.py '
              '--ratio 0.1 '
              '--division 8 '
              '--cur_iter 1 '
              '--num_iters 5 ')


