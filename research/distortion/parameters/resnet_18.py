from research.parameters.base import BLOCK_SIZES_96x96, BLOCK_SIZES_192x192


class ResNet18_Params:
    def __init__(self):
        self.BACKBONE = "AvgPoolResNet"
        self.BLOCK_NAMES = [
            'stem',
            'layer1_0',
            'layer1_1',
            'layer2_0',
            'layer2_1',
            'layer3_0',
            'layer3_1',
            'layer4_0',
            'layer4_1',
            'decode',
            None
        ]

        self.LAYER_NAMES = \
            [
                "stem_2",
                "stem_5",
                "stem_8",
                "layer1_0_1",
                "layer1_0_2",
                "layer1_1_1",
                "layer1_1_2",
                "layer2_0_1",
                "layer2_0_2",
                "layer2_1_1",
                "layer2_1_2",
                "layer3_0_1",
                "layer3_0_2",
                "layer3_1_1",
                "layer3_1_2",
                "layer4_0_1",
                "layer4_0_2",
                "layer4_1_1",
                "layer4_1_2",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                "stem_2": "stem",
                "stem_5": "stem",
                "stem_8": "stem",
                "layer1_0_1": "layer1_0",
                "layer1_0_2": "layer1_0",
                "layer1_1_1": "layer1_1",
                "layer1_1_2": "layer1_1",
                "layer2_0_1": "layer2_0",
                "layer2_0_2": "layer2_0",
                "layer2_1_1": "layer2_1",
                "layer2_1_2": "layer2_1",
                "layer3_0_1": "layer3_0",
                "layer3_0_2": "layer3_0",
                "layer3_1_1": "layer3_1",
                "layer3_1_2": "layer3_1",
                "layer4_0_1": "layer4_0",
                "layer4_0_2": "layer4_0",
                "layer4_1_1": "layer4_1",
                "layer4_1_2": "layer4_1",
                "decode_0": "decode",
                "decode_1": "decode",
                "decode_2": "decode",
                "decode_3": "decode",
                "decode_4": "decode",
                "decode_5": "decode",
            }

        self.BLOCK_INPUT_DICT = \
            {
                "stem": "input_images",
                "layer1_0": "stem",
                "layer1_1": "layer1_0",
                "layer2_0": "layer1_1",
                "layer2_1": "layer2_0",
                "layer3_0": "layer2_1",
                "layer3_1": "layer3_0",
                "layer4_0": "layer3_1",
                "layer4_1": "layer4_0",
                "decode": "layer4_1",
                None: "decode"
            }

        self.IN_LAYER_PROXY_SPEC = \
            {
                'stem_2': 'decode',
                'stem_5': 'decode',
                'stem_8': 'decode',
                'layer1_0_1': 'decode',
                'layer1_0_2': 'decode',
                'layer1_1_1': 'decode',
                'layer1_1_2': 'decode',
                'layer2_0_1': 'decode',
                'layer2_0_2': 'decode',
                'layer2_1_1': 'decode',
                'layer2_1_2': 'decode',
                'layer3_0_1': 'decode',
                'layer3_0_2': 'decode',
                'layer3_1_1': 'decode',
                'layer3_1_2': 'decode',
                'layer4_0_1': 'decode',
                'layer4_0_2': 'decode',
                'layer4_1_1': 'decode',
                'layer4_1_2': 'decode',
                'decode_0': 'decode',
                'decode_1': 'decode',
                'decode_2': 'decode',
                'decode_3': 'decode',
                'decode_4': 'decode',
                'decode_5': 'decode'
            }

class ResNet18_Params_96x96(ResNet18_Params):
    def __init__(self):
        super().__init__()
        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'stem_2': BLOCK_SIZES_96x96,
                'stem_5': BLOCK_SIZES_96x96,
                'stem_8': BLOCK_SIZES_96x96,
                'layer1_0_1': BLOCK_SIZES_96x96,
                'layer1_0_2': BLOCK_SIZES_96x96,
                'layer1_1_1': BLOCK_SIZES_96x96,
                'layer1_1_2': BLOCK_SIZES_96x96,
                'layer2_0_1': BLOCK_SIZES_96x96,
                'layer2_0_2': BLOCK_SIZES_96x96,
                'layer2_1_1': BLOCK_SIZES_96x96,
                'layer2_1_2': BLOCK_SIZES_96x96,
                'layer3_0_1': BLOCK_SIZES_96x96,
                'layer3_0_2': BLOCK_SIZES_96x96,
                'layer3_1_1': BLOCK_SIZES_96x96,
                'layer3_1_2': BLOCK_SIZES_96x96,
                'layer4_0_1': BLOCK_SIZES_96x96,
                'layer4_0_2': BLOCK_SIZES_96x96,
                'layer4_1_1': BLOCK_SIZES_96x96,
                'layer4_1_2': BLOCK_SIZES_96x96,
                'decode_0': BLOCK_SIZES_96x96,
                'decode_1': BLOCK_SIZES_96x96,
                'decode_2': BLOCK_SIZES_96x96,
                'decode_3': BLOCK_SIZES_96x96,
                'decode_4': BLOCK_SIZES_96x96,
                'decode_5': BLOCK_SIZES_96x96
            }

        self.LAYER_NAME_TO_DIMS = {
            "stem_2": [32, 48, 48],
            "stem_5": [32, 48, 48],
            "stem_8": [64, 48, 48],

            "layer1_0_1": [64, 24, 24],
            "layer1_0_2": [64, 24, 24],
            "layer1_1_1": [64, 24, 24],
            "layer1_1_2": [64, 24, 24],

            "layer2_0_1": [128, 12, 12],
            "layer2_0_2": [128, 12, 12],
            "layer2_1_1": [128, 12, 12],
            "layer2_1_2": [128, 12, 12],

            "layer3_0_1": [256, 12, 12],
            "layer3_0_2": [256, 12, 12],
            "layer3_1_1": [256, 12, 12],
            "layer3_1_2": [256, 12, 12],

            "layer4_0_1": [512, 12, 12],
            "layer4_0_2": [512, 12, 12],
            "layer4_1_1": [512, 12, 12],
            "layer4_1_2": [512, 12, 12],

            "decode_0": [128, 1, 1],
            "decode_1": [128, 12, 12],
            "decode_2": [128, 12, 12],
            "decode_3": [128, 12, 12],
            "decode_4": [128, 12, 12],
            "decode_5": [128, 12, 12],
        }


class ResNet18_Params_192x192(ResNet18_Params):
    def __init__(self):
        super().__init__()
        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'stem_2': BLOCK_SIZES_192x192,
                'stem_5': BLOCK_SIZES_192x192,
                'stem_8': BLOCK_SIZES_192x192,
                'layer1_0_1': BLOCK_SIZES_192x192,
                'layer1_0_2': BLOCK_SIZES_192x192,
                'layer1_1_1': BLOCK_SIZES_192x192,
                'layer1_1_2': BLOCK_SIZES_192x192,
                'layer2_0_1': BLOCK_SIZES_192x192,
                'layer2_0_2': BLOCK_SIZES_192x192,
                'layer2_1_1': BLOCK_SIZES_192x192,
                'layer2_1_2': BLOCK_SIZES_192x192,
                'layer3_0_1': BLOCK_SIZES_192x192,
                'layer3_0_2': BLOCK_SIZES_192x192,
                'layer3_1_1': BLOCK_SIZES_192x192,
                'layer3_1_2': BLOCK_SIZES_192x192,
                'layer4_0_1': BLOCK_SIZES_192x192,
                'layer4_0_2': BLOCK_SIZES_192x192,
                'layer4_1_1': BLOCK_SIZES_192x192,
                'layer4_1_2': BLOCK_SIZES_192x192,
                'decode_0': BLOCK_SIZES_192x192,
                'decode_1': BLOCK_SIZES_192x192,
                'decode_2': BLOCK_SIZES_192x192,
                'decode_3': BLOCK_SIZES_192x192,
                'decode_4': BLOCK_SIZES_192x192,
                'decode_5': BLOCK_SIZES_192x192
            }

        self.LAYER_NAME_TO_DIMS = {
            "stem_2": [32, 96, 96],
            "stem_5": [32, 96, 96],
            "stem_8": [64, 96, 96],

            "layer1_0_1": [64, 48, 48],
            "layer1_0_2": [64, 48, 48],
            "layer1_1_1": [64, 48, 48],
            "layer1_1_2": [64, 48, 48],

            "layer2_0_1": [128, 24, 24],
            "layer2_0_2": [128, 24, 24],
            "layer2_1_1": [128, 24, 24],
            "layer2_1_2": [128, 24, 24],

            "layer3_0_1": [256, 24, 24],
            "layer3_0_2": [256, 24, 24],
            "layer3_1_1": [256, 24, 24],
            "layer3_1_2": [256, 24, 24],

            "layer4_0_1": [512, 24, 24],
            "layer4_0_2": [512, 24, 24],
            "layer4_1_1": [512, 24, 24],
            "layer4_1_2": [512, 24, 24],

            "decode_0": [128, 1, 1],
            "decode_1": [128, 24, 24],
            "decode_2": [128, 24, 24],
            "decode_3": [128, 24, 24],
            "decode_4": [128, 24, 24],
            "decode_5": [128, 24, 24],
        }
