from research.distortion.parameters.mobile_net import MobileNetV2_256_Params_2_Groups, MobileNetV2_512_Params_2_Groups, MobileNetV2_256_Params_1_Groups, MobileNetV2_256_Params_1_Groups_mini
from research.distortion.parameters.resnet_18 import ResNet18_Params_96x96, ResNet18_Params_192x192
# from research.distortion.parameters.classification.ResNet_CIFAR import ResNet18_CIFAR, ResNet50_CIFAR
from research.distortion.parameters.segmentation.MobileNetV2 import MobileNetV2
from research.distortion.parameters.classification.resent.resnet18_8xb32_in1k import Params as resnet18_8xb32_in1k_Params
class ParamsFactory:
    def __init__(self):
        pass

    def __call__(self, cfg):
        if cfg.model.type == 'ImageClassifier' and cfg.model.backbone.type == 'MyResNet' and cfg.model.backbone.depth == 18:
            return resnet18_8xb32_in1k_Params()

        if cfg.type == 'ResNet_CIFAR_V2':
            if cfg.depth == 18:
                return ResNet18_CIFAR()
            if cfg.depth == 50:
                return ResNet50_CIFAR()
        if cfg.type == 'MobileNetV2':
            return MobileNetV2()
        else:
            assert False, "Unknown model type: {}".format(cfg.type)

param_factory = ParamsFactory()