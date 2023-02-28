_base_ = [
    '../../_base_/models/resnet50_avg_pool.py', '../../_base_/datasets/imagenet_bs64.py',
    '../../_base_/schedules/imagenet_bs256_finetune.py', '../../_base_/default_runtime.py'
]
