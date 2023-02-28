_base_ = [
    '../../_base_/datasets/cifar100_bs64.py',
    '../../_base_/schedules/cifar10_bs128.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.004)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)

# checkpoint saving
checkpoint_config = dict(interval=10)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='gloo')
log_level = 'INFO'
load_from = None
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1)]