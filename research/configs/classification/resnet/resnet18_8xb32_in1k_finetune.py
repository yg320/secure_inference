_base_ = [
    '../_base_/models/resnet18_avg_pool.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_finetune_v2.py', '../_base_/default_runtime.py'
]

relu_spec_file = None
load_from = "./mmlab_models/classification/resnet18_8xb32_in1k_20210831-fbbb1da6.pth"
work_dir = "./outputs/classification/resnet18_8xb32_in1k/finetune"
