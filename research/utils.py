from mmcls.datasets import build_dataset as build_dataset_mmcls
from mmseg.datasets import build_dataset as build_dataset_mmseg

def build_data(cfg, train):

    data_cfg = cfg.data.train if train else cfg.data.test
    if cfg.model.type == 'ImageClassifier':
        dataset = build_dataset_mmcls(data_cfg, default_args=dict(test_mode=False))
    elif cfg.model.type == 'EncoderDecoder':
        dataset = build_dataset_mmseg(data_cfg, default_args=dict(test_mode=False))
    else:
        raise NotImplementedError

    return dataset