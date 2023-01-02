from research.distortion.arch_utils.classification.resnet.resnet18_8xb32_in1k import Utils as resnet18_8xb32_in1k_Utils
from research.distortion.arch_utils.classification.resnet.resnet50_8xb32_in1k import Utils as resnet50_8xb32_in1k_Utils
from research.distortion.arch_utils.segmentation.MobileNetV2 import MobileNetV2_Utils as MobileNetV2_Segmentation_Utils


class ArchUtilsFactory:
    def __call__(self, cfg):

        if cfg.model.type == 'ImageClassifier' and cfg.model.backbone.type in ['MyResNet', 'AvgPoolResNet']:
            # TODO: we should call resnet_utils(depth=cfg.model.backbone.depth), as there is a lot of code duplication.
            # TODO: moreover, this code should be general. (If file names are consistent, we can just import the file and call the class)

            if cfg.model.backbone.depth == 18:
                return resnet18_8xb32_in1k_Utils()
            elif cfg.model.backbone.depth == 50:
                return resnet50_8xb32_in1k_Utils()
        if cfg.model.type == 'EncoderDecoder' and cfg.model.backbone.type in ['MobileNetV2']:
            return MobileNetV2_Segmentation_Utils()


arch_utils_factory = ArchUtilsFactory()