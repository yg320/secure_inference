from research.distortion.arch_utils.classification.resnet.resnet50_8xb32_in1k import ResNetUtils as \
    resnet50_8xb32_in1k_Utils
from research.distortion.arch_utils.classification.resnet.resnet18_cifar import ResNet18_CIFAR_Utils
from research.distortion.arch_utils.segmentation.MobileNetV2 import MobileNetV2_Utils as MobileNetV2_Segmentation_Utils
from research.distortion.arch_utils.segmentation.resnet50 import ResNetUtils as ResNet50_Segmentation_Utils


class ArchUtilsFactory:
    def __call__(self, cfg):

        if cfg.model.type == 'ImageClassifier' and cfg.model.backbone.type in ['MyResNet', 'AvgPoolResNet']:
            if cfg.model.backbone.depth == 50:
                return resnet50_8xb32_in1k_Utils()
        if cfg.model.type == 'ImageClassifier' and "ResNet_CIFAR_V2" in cfg.model.backbone.type:
            if cfg.model.backbone.depth == 18:
                return ResNet18_CIFAR_Utils()
        if cfg.model.type == 'EncoderDecoder' and cfg.model.backbone.type in ['MobileNetV2']:
            return MobileNetV2_Segmentation_Utils()
        if cfg.model.type == 'EncoderDecoder' and cfg.model.backbone.type in ['AvgPoolResNetSeg']:
            return ResNet50_Segmentation_Utils()

        raise NotImplementedError


arch_utils_factory = ArchUtilsFactory()
