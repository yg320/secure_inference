_base_ = [
    '../_base_/datasets/cifar100_bs64.py',
    '../_base_/schedules/cifar10_bs128.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2_conv_stride_2',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        # stem_channels=32,
        # base_channels=32,
        # strides=(2, 2, 2, 2),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0004)
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=120)


relu_spec_file = "/home/yakir/tesnet18/distortions/resnet18_2xb64_cifar100/block_sizes/S2RT_S3RT_alpha_0.5_rho_0.5_version_0.pickle"
