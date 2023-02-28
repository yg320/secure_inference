_base_ = './pascal_voc12_2x8.py'
# dataset settings
data = dict(
    train=dict(
        ann_dir=['SegmentationClass', 'SegmentationClassAug'],
        split=[
            'ImageSets/Segmentation/train.txt',
            'ImageSets/Segmentation/aug.txt'
        ]))