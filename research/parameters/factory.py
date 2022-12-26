from research.parameters.mobile_net import MobileNetV2_256_Params_2_Groups, MobileNetV2_512_Params_2_Groups, MobileNetV2_256_Params_1_Groups, MobileNetV2_256_Params_1_Groups_mini
from research.parameters.resnet_18 import ResNet18_Params_96x96, ResNet18_Params_192x192

from research.parameters.classification.ResNet_CIFAR import ResNet18_CIFAR, ResNet50_CIFAR

class ParamsFactory:
    def __init__(self):
        pass

    def __call__(self, model_cfg):
        if model_cfg.type == 'ResNet_CIFAR_V2':
            if model_cfg.depth == 18:
                return ResNet18_CIFAR()
            if model_cfg.depth == 50:
                return ResNet50_CIFAR()
        else:
            assert False, "Unknown model type: {}".format(model_cfg.type)
        # return self.classes[type_]()

param_factory = ParamsFactory()