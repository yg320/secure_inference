_base_ = [
    '../../_base_/datasets/cifar100_bs64.py', '../../_base_/schedules/cifar10_bs128.py', '../../_base_/default_runtime.py'
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

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.001)
lr_config = dict(policy='step', step=[30, 60, 90], gamma=0.25, min_lr=0.0001, warmup='linear', warmup_ratio=0.01, warmup_iters=5, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=120)



relu_spec_file = "/home/yakir/deepreduce_comparison_v3/distortions/baseline/block_sizes/4x4_algo.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/baseline/epoch_200.pth"


dist_params = dict(backend='gloo')
log_level = 'INFO'
resume_from = None
evaluation = dict(interval=10, by_epoch=True)
workflow = [('train', 10), ('val', 1)]