_base_ = [
    '../../_base_/models/resnet50_avg_pool.py', '../../_base_/datasets/imagenet_bs64.py',
    '../../_base_/schedules/imagenet_bs256_finetune.py', '../../_base_/default_runtime.py'
]

load_from = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
work_dir = "./outputs_v2/experiments/classification/resnet50_8xb32_in1k/benchmark/baseline"



