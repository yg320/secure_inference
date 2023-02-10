_base_ = [
    '../../_base_/models/resnet50_avg_pool.py', '../../_base_/datasets/imagenet_bs64.py',
    '../../_base_/schedules/imagenet_bs256_finetune.py', '../../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[50], gamma=0.1, min_lr=0.0001, warmup='linear', warmup_ratio=0.01, warmup_iters=1, warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=25)