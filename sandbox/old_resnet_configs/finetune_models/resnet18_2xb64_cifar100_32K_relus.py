_base_ = [
    '../../_base_/models/resnet18_cifar.py', '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[25, 35, 45], gamma=0.25, min_lr=0.0001, warmup='linear', warmup_ratio=0.01, warmup_iters=1, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=50)


relu_spec_file = "/home/yakir/distortion_200/block_size/32K.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/resnet18_2xb64_cifar100/epoch_200.pth"


