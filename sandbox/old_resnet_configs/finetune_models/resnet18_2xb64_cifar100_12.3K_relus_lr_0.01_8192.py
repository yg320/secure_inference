_base_ = [
    '../../_base_/models/resnet18_cifar.py', '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.2, min_lr=0.0001, warmup='linear', warmup_ratio=0.01, warmup_iters=5, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)


relu_spec_file = "/home/yakir/epoch_199_distortions/block_size/12.3K.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/resnet18_2xb64_cifar100/epoch_199.pth"


