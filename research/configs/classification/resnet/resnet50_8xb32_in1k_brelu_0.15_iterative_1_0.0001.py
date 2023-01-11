_base_ = [
    '../_base_/models/resnet50_avg_pool.py', '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_finetune_0.0001.py', '../_base_/default_runtime.py'
]

relu_spec_file = "./relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_1/block_size_spec_0.1.pickle"
load_from = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_13.pth"
work_dir = "./outputs/classification/resnet50_8xb32_in1k/brelu_0.15_iterative_0.0001/iter_1"
