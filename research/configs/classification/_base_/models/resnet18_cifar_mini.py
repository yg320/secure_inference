# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR_V2_mini',
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
        num_classes=10,
        in_channels=384,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
