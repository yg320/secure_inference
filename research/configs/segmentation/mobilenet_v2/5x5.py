_base_ = '../deeplabv3/deeplabv3_r101-d8_512x512_160k_ade20k.py'
model = dict(
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        _delete_=True,
        type='MobileNetV2',
        widen_factor=1.,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6),
        act_cfg=dict(type='ReLU'),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    decode_head=dict(in_channels=320),
    auxiliary_head=dict(in_channels=96))

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)

relu_spec_file = "./relu_spec_files/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/5x5.pickle"
load_from = "./mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth"
work_dir = "./outputs_v2/experiments/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k/5x5"