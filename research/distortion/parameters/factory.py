from research.distortion.parameters.classification.resent.resnet18_8xb32_in1k import Params as resnet18_8xb32_in1k_Params
from research.distortion.parameters.classification.resent.resnet50_8xb32_in1k import Params as resnet50_8xb32_in1k_Params

from research.distortion.parameters.segmentation.MobileNetV2 import Params as MobileNetV2_Params


class ParamsFactory:
    def __init__(self):
        pass

    def __call__(self, cfg):
        if cfg.model.type == 'ImageClassifier' and cfg.model.backbone.type in ['MyResNet', 'AvgPoolResNet']:
            if cfg.model.backbone.depth == 18:
                return resnet18_8xb32_in1k_Params()
            elif cfg.model.backbone.depth == 50:
                return resnet50_8xb32_in1k_Params()
        elif cfg.model.type == 'EncoderDecoder' and cfg.model.backbone.type == 'MobileNetV2':
            return MobileNetV2_Params()
        raise NotImplementedError
param_factory = ParamsFactory()