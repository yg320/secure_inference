from research.parameters.mobile_net import MobileNetV2_256_Params_2_Groups, MobileNetV2_512_Params_2_Groups, MobileNetV2_256_Params_1_Groups, MobileNetV2_256_Params_1_Groups_mini
from research.parameters.resnet_18 import ResNet18_Params_96x96, ResNet18_Params_192x192


class ParamsFactory:
    def __init__(self):
        self.classes = {
            "MobileNetV2_256_Params_2_Groups": MobileNetV2_256_Params_2_Groups,
            "MobileNetV2_512_Params_2_Groups": MobileNetV2_512_Params_2_Groups,
            "MobileNetV2_256_Params_1_Groups": MobileNetV2_256_Params_1_Groups,
            "MobileNetV2_256_Params_1_Groups_mini": MobileNetV2_256_Params_1_Groups_mini,
            "ResNet18_Params_96x96": ResNet18_Params_96x96,
            "ResNet18_Params_192x192": ResNet18_Params_192x192,
        }

    def __call__(self, type_):
        return self.classes[type_]()
