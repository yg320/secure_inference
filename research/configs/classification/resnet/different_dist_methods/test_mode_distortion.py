_base_ = [
    '../../_base_/models/resnet50_avg_pool.py', '../../_base_/datasets/imagenet_bs64.py',
    '../../_base_/schedules/imagenet_bs256_finetune_0.005_15_epochs.py', '../../_base_/default_runtime.py'
]

relu_spec_file = "./relu_spec_files_v2/classification/resnet50_8xb32_in1k/test_mode_distortion_0.1.pickle"
load_from = "./outputs/classification/resnet50_8xb32_in1k/finetune_0.0001_avg_pool/epoch_14.pth"
work_dir = "./outputs_v2/distortions/classification/resnet50_8xb32_in1k/different_dist_methods/test_mode_distortion"



