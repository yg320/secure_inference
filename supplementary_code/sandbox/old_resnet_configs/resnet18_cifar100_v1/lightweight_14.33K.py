_base_ = [
    '../../_base_/datasets/cifar100_bs64.py', '../../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2_lightweight',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0004)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='Cyclic',
    cyclic_times=2,
    target_ratio=(10, 1e-4),
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=120)

relu_spec_file = "/home/yakir/deepreduce_comparison_v2/distortions/lightweight/block_sizes/14.33K.pickle"
load_from = "/home/yakir/PycharmProjects/secure_inference/work_dirs/lightweight/epoch_200.pth"


