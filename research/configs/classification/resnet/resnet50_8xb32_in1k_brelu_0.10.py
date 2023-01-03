_base_ = [
    '../_base_/models/resnet50_avg_pool.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_finetune.py', '../_base_/default_runtime.py'
]

relu_spec_file = "./relu_spec_files/classification/resnet50_8xb32_in1k/brelu_0.10.pickle"
load_from = "./mmlab_models/classification/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
work_dir = "./outputs/classification/resnet50_8xb32_in1k/brelu_0.10"
