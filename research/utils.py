from mmcls.datasets import build_dataset as build_dataset_mmcls
from mmseg.datasets import build_dataset as build_dataset_mmseg
# from mmdet.datasets import build_dataset as build_dataset_mmdet
def build_data(cfg, mode='train'):

    if mode == 'train':
        data_cfg = cfg.data.train
    elif mode == 'test':
        data_cfg = cfg.data.test
    elif mode == 'distortion_extraction':
        data_cfg = cfg.data.distortion_extraction
    else:
        raise ValueError(f'Unknown mode {mode}')

    if cfg.model.type == 'ImageClassifier':
        dataset = build_dataset_mmcls(data_cfg, default_args=dict(test_mode=False))
    elif cfg.model.type == 'EncoderDecoder':
        dataset = build_dataset_mmseg(data_cfg, default_args=dict(test_mode=False))
    elif cfg.model.type == 'SingleStageDetector':
        dataset = build_dataset_mmdet(data_cfg, default_args=dict(test_mode=False))
    else:
        raise NotImplementedError

    return dataset