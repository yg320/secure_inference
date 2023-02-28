_base_ = [
    '../_base_/models/resnet18_cifar.py', '../_base_/datasets/cifar100_bs64.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

model = dict(head=dict(num_classes=100))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0004)
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=120)


relu_spec_file = "/home/yakir/tesnet18/distortions/resnet18_2xb64_cifar100/block_sizes/S1.pickle"
