_base_ = [
    '../../_base_/models/resnet18_cifar.py', '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2, warmup='linear', warmup_ratio=0.01, warmup_iters=5, warmup_by_epoch=True)

relu_spec_file = "/home/yakir/epoch_199_distortions/block_size/14.33K.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/resnet18_2xb64_cifar100_14.3K_relus/epoch_100.pth"


