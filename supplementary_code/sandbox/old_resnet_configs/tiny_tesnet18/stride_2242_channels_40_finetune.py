_base_ = [
     '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2_mini_stride2242_channels40',
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
        in_channels=320,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.25, min_lr=0.0001, warmup='linear', warmup_ratio=0.01, warmup_iters=5, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)


relu_spec_file = "/home/yakir/tiny_tesnet18/distortions/stride_2242_channels_40/14.33K.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/stride_2242_channels_40/epoch_200.pth"


