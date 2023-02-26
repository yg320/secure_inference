_base_ = [
    '../_base_/models/deeplabv3_r50-d8_avg_pool.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

# optimizer
optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=15000)
checkpoint_config = dict(by_epoch=False, interval=3000)
evaluation = dict(interval=3000, metric='mIoU', pre_eval=True)