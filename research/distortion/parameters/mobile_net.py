from research.distortion.parameters.base import BLOCK_SIZES_FULL, BLOCK_SIZES_MINI, BLOCK_SIZES_256x256

class MobileNetV2_Params:
    def __init__(self):
        self.BACKBONE = "MobileNetV2"

        self.BLOCK_NAMES = \
            [
                "conv1",
                "layer1_0",
                "layer2_0",
                "layer2_1",
                "layer3_0",
                "layer3_1",
                "layer3_2",
                "layer4_0",
                "layer4_1",
                "layer4_2",
                "layer4_3",
                "layer5_0",
                "layer5_1",
                "layer5_2",
                "layer6_0",
                "layer6_1",
                "layer6_2",
                "layer7_0",
                "decode",
                None
            ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                'conv1': 'conv1',
                'layer1_0_0': 'layer1_0',
                'layer2_0_0': 'layer2_0',
                'layer2_0_1': 'layer2_0',
                'layer2_1_0': 'layer2_1',
                'layer2_1_1': 'layer2_1',
                'layer3_0_0': 'layer3_0',
                'layer3_0_1': 'layer3_0',
                'layer3_1_0': 'layer3_1',
                'layer3_1_1': 'layer3_1',
                'layer3_2_0': 'layer3_2',
                'layer3_2_1': 'layer3_2',
                'layer4_0_0': 'layer4_0',
                'layer4_0_1': 'layer4_0',
                'layer4_1_0': 'layer4_1',
                'layer4_1_1': 'layer4_1',
                'layer4_2_0': 'layer4_2',
                'layer4_2_1': 'layer4_2',
                'layer4_3_0': 'layer4_3',
                'layer4_3_1': 'layer4_3',
                'layer5_0_0': 'layer5_0',
                'layer5_0_1': 'layer5_0',
                'layer5_1_0': 'layer5_1',
                'layer5_1_1': 'layer5_1',
                'layer5_2_0': 'layer5_2',
                'layer5_2_1': 'layer5_2',
                'layer6_0_0': 'layer6_0',
                'layer6_0_1': 'layer6_0',
                'layer6_1_0': 'layer6_1',
                'layer6_1_1': 'layer6_1',
                'layer6_2_0': 'layer6_2',
                'layer6_2_1': 'layer6_2',
                'layer7_0_0': 'layer7_0',
                'layer7_0_1': 'layer7_0',
                'decode_0': 'decode',
                'decode_1': 'decode',
                'decode_2': 'decode',
                'decode_3': 'decode',
                'decode_4': 'decode',
                'decode_5': 'decode'
            }

        self.LAYER_NAMES = \
            [
                "conv1",
                "layer1_0_0",
                "layer2_0_0",
                "layer2_0_1",
                "layer2_1_0",
                "layer2_1_1",
                "layer3_0_0",
                "layer3_0_1",
                "layer3_1_0",
                "layer3_1_1",
                "layer3_2_0",
                "layer3_2_1",
                "layer4_0_0",
                "layer4_0_1",
                "layer4_1_0",
                "layer4_1_1",
                "layer4_2_0",
                "layer4_2_1",
                "layer4_3_0",
                "layer4_3_1",
                "layer5_0_0",
                "layer5_0_1",
                "layer5_1_0",
                "layer5_1_1",
                "layer5_2_0",
                "layer5_2_1",
                "layer6_0_0",
                "layer6_0_1",
                "layer6_1_0",
                "layer6_1_1",
                "layer6_2_0",
                "layer6_2_1",
                "layer7_0_0",
                "layer7_0_1",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]

        self.BLOCK_INPUT_DICT = \
            {
                "conv1": "input_images",
                "layer1_0": "conv1",
                "layer2_0": "layer1_0",
                "layer2_1": "layer2_0",
                "layer3_0": "layer2_1",
                "layer3_1": "layer3_0",
                "layer3_2": "layer3_1",
                "layer4_0": "layer3_2",
                "layer4_1": "layer4_0",
                "layer4_2": "layer4_1",
                "layer4_3": "layer4_2",
                "layer5_0": "layer4_3",
                "layer5_1": "layer5_0",
                "layer5_2": "layer5_1",
                "layer6_0": "layer5_2",
                "layer6_1": "layer6_0",
                "layer6_2": "layer6_1",
                "layer7_0": "layer6_2",
                "decode": "layer7_0",
                None: "decode"
            }

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'conv1': BLOCK_SIZES_FULL,
                'layer1_0_0': BLOCK_SIZES_FULL,
                'layer2_0_0': BLOCK_SIZES_FULL,
                'layer2_0_1': BLOCK_SIZES_FULL,
                'layer2_1_0': BLOCK_SIZES_FULL,
                'layer2_1_1': BLOCK_SIZES_FULL,
                'layer3_0_0': BLOCK_SIZES_FULL,
                'layer3_0_1': BLOCK_SIZES_FULL,
                'layer3_1_0': BLOCK_SIZES_FULL,
                'layer3_1_1': BLOCK_SIZES_FULL,
                'layer3_2_0': BLOCK_SIZES_FULL,
                'layer3_2_1': BLOCK_SIZES_FULL,
                'layer4_0_0': BLOCK_SIZES_FULL,
                'layer4_0_1': BLOCK_SIZES_FULL,
                'layer4_1_0': BLOCK_SIZES_FULL,
                'layer4_1_1': BLOCK_SIZES_FULL,
                'layer4_2_0': BLOCK_SIZES_FULL,
                'layer4_2_1': BLOCK_SIZES_FULL,
                'layer4_3_0': BLOCK_SIZES_FULL,
                'layer4_3_1': BLOCK_SIZES_FULL,
                'layer5_0_0': BLOCK_SIZES_FULL,
                'layer5_0_1': BLOCK_SIZES_FULL,
                'layer5_1_0': BLOCK_SIZES_FULL,
                'layer5_1_1': BLOCK_SIZES_FULL,
                'layer5_2_0': BLOCK_SIZES_FULL,
                'layer5_2_1': BLOCK_SIZES_FULL,
                'layer6_0_0': BLOCK_SIZES_FULL,
                'layer6_0_1': BLOCK_SIZES_FULL,
                'layer6_1_0': BLOCK_SIZES_FULL,
                'layer6_1_1': BLOCK_SIZES_FULL,
                'layer6_2_0': BLOCK_SIZES_FULL,
                'layer6_2_1': BLOCK_SIZES_FULL,
                'layer7_0_0': BLOCK_SIZES_FULL,
                'layer7_0_1': BLOCK_SIZES_FULL,
                'decode_0': BLOCK_SIZES_MINI,
                'decode_1': BLOCK_SIZES_MINI,
                'decode_2': BLOCK_SIZES_MINI,
                'decode_3': BLOCK_SIZES_MINI,
                'decode_4': BLOCK_SIZES_MINI,
                'decode_5': BLOCK_SIZES_MINI
            }

class MobileNetV2_256_Params(MobileNetV2_Params):
    def __init__(self):
        super().__init__()

        self.LAYER_NAME_TO_DIMS = {'conv1': [32, 128, 128],
                                   'layer1_0_0': [32, 128, 128],
                                   'layer2_0_0': [96, 128, 128],
                                   'layer2_0_1': [96, 64, 64],
                                   'layer2_1_0': [144, 64, 64],
                                   'layer2_1_1': [144, 64, 64],
                                   'layer3_0_0': [144, 64, 64],
                                   'layer3_0_1': [144, 32, 32],
                                   'layer3_1_0': [192, 32, 32],
                                   'layer3_1_1': [192, 32, 32],
                                   'layer3_2_0': [192, 32, 32],
                                   'layer3_2_1': [192, 32, 32],
                                   'layer4_0_0': [192, 32, 32],
                                   'layer4_0_1': [192, 32, 32],
                                   'layer4_1_0': [384, 32, 32],
                                   'layer4_1_1': [384, 32, 32],
                                   'layer4_2_0': [384, 32, 32],
                                   'layer4_2_1': [384, 32, 32],
                                   'layer4_3_0': [384, 32, 32],
                                   'layer4_3_1': [384, 32, 32],
                                   'layer5_0_0': [384, 32, 32],
                                   'layer5_0_1': [384, 32, 32],
                                   'layer5_1_0': [576, 32, 32],
                                   'layer5_1_1': [576, 32, 32],
                                   'layer5_2_0': [576, 32, 32],
                                   'layer5_2_1': [576, 32, 32],
                                   'layer6_0_0': [576, 32, 32],
                                   'layer6_0_1': [576, 32, 32],
                                   'layer6_1_0': [960, 32, 32],
                                   'layer6_1_1': [960, 32, 32],
                                   'layer6_2_0': [960, 32, 32],
                                   'layer6_2_1': [960, 32, 32],
                                   'layer7_0_0': [960, 32, 32],
                                   'layer7_0_1': [960, 32, 32],
                                   'decode_0':   [512, 1, 1],
                                   'decode_1':   [512, 32, 32],
                                   'decode_2':   [512, 32, 32],
                                   'decode_3':   [512, 32, 32],
                                   'decode_4':   [512, 32, 32],
                                   'decode_5':   [512, 32, 32]}

class MobileNetV2_512_Params(MobileNetV2_Params):
    def __init__(self):
        super().__init__()

        self.LAYER_NAME_TO_DIMS = {'conv1': [32, 256, 256],
                                   'layer1_0_0': [32, 256, 256],
                                   'layer2_0_0': [96, 256, 256],
                                   'layer2_0_1': [96, 128, 128],
                                   'layer2_1_0': [144, 128, 128],
                                   'layer2_1_1': [144, 128, 128],
                                   'layer3_0_0': [144, 128, 128],
                                   'layer3_0_1': [144, 64, 64],
                                   'layer3_1_0': [192, 64, 64],
                                   'layer3_1_1': [192, 64, 64],
                                   'layer3_2_0': [192, 64, 64],
                                   'layer3_2_1': [192, 64, 64],
                                   'layer4_0_0': [192, 64, 64],
                                   'layer4_0_1': [192, 64, 64],
                                   'layer4_1_0': [384, 64, 64],
                                   'layer4_1_1': [384, 64, 64],
                                   'layer4_2_0': [384, 64, 64],
                                   'layer4_2_1': [384, 64, 64],
                                   'layer4_3_0': [384, 64, 64],
                                   'layer4_3_1': [384, 64, 64],
                                   'layer5_0_0': [384, 64, 64],
                                   'layer5_0_1': [384, 64, 64],
                                   'layer5_1_0': [576, 64, 64],
                                   'layer5_1_1': [576, 64, 64],
                                   'layer5_2_0': [576, 64, 64],
                                   'layer5_2_1': [576, 64, 64],
                                   'layer6_0_0': [576, 64, 64],
                                   'layer6_0_1': [576, 64, 64],
                                   'layer6_1_0': [960, 64, 64],
                                   'layer6_1_1': [960, 64, 64],
                                   'layer6_2_0': [960, 64, 64],
                                   'layer6_2_1': [960, 64, 64],
                                   'layer7_0_0': [960, 64, 64],
                                   'layer7_0_1': [960, 64, 64],
                                   'decode_0': [512, 1, 1],
                                   'decode_1': [512, 64, 64],
                                   'decode_2': [512, 64, 64],
                                   'decode_3': [512, 64, 64],
                                   'decode_4': [512, 64, 64],
                                   'decode_5': [512, 64, 64]}

class MobileNetV2_256_Params_2_Groups(MobileNetV2_256_Params):
    def __init__(self):

        super().__init__()
        self.LAYER_GROUPS = \
            [
                [
                    "conv1",
                    "layer1_0_0",
                    "layer2_0_0",
                    "layer2_0_1",
                    "layer2_1_0",
                    "layer2_1_1",
                    "layer3_0_0",
                    "layer3_0_1",
                    "layer3_1_0",
                    "layer3_1_1",
                    "layer3_2_0",
                    "layer3_2_1",
                    "layer4_0_0",
                    "layer4_0_1",
                    "layer4_1_0",
                    "layer4_1_1",
                    "layer4_2_0",
                    "layer4_2_1",
                    "layer4_3_0",
                    "layer4_3_1",
                    "layer5_0_0",
                    "layer5_0_1",
                    "layer5_1_0",
                    "layer5_1_1",
                    "layer5_2_0",
                    "layer5_2_1",
                    "layer6_0_0",
                    "layer6_0_1",
                    "layer6_1_0",
                    "layer6_1_1",
                    "layer6_2_0",
                    "layer6_2_1",
                    "layer7_0_0",
                    "layer7_0_1",
                ],
                [
                    "decode_0",
                    "decode_1",
                    "decode_2",
                    "decode_3",
                    "decode_4",
                    "decode_5",
                ]
            ]
        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1':      'layer7_0',
                'layer1_0_0': 'layer7_0',
                'layer2_0_0': 'layer7_0',
                'layer2_0_1': 'layer7_0',
                'layer2_1_0': 'layer7_0',
                'layer2_1_1': 'layer7_0',
                'layer3_0_0': 'layer7_0',
                'layer3_0_1': 'layer7_0',
                'layer3_1_0': 'layer7_0',
                'layer3_1_1': 'layer7_0',
                'layer3_2_0': 'layer7_0',
                'layer3_2_1': 'layer7_0',
                'layer4_0_0': 'layer7_0',
                'layer4_0_1': 'layer7_0',
                'layer4_1_0': 'layer7_0',
                'layer4_1_1': 'layer7_0',
                'layer4_2_0': 'layer7_0',
                'layer4_2_1': 'layer7_0',
                'layer4_3_0': 'layer7_0',
                'layer4_3_1': 'layer7_0',
                'layer5_0_0': 'layer7_0',
                'layer5_0_1': 'layer7_0',
                'layer5_1_0': 'layer7_0',
                'layer5_1_1': 'layer7_0',
                'layer5_2_0': 'layer7_0',
                'layer5_2_1': 'layer7_0',
                'layer6_0_0': 'layer7_0',
                'layer6_0_1': 'layer7_0',
                'layer6_1_0': 'layer7_0',
                'layer6_1_1': 'layer7_0',
                'layer6_2_0': 'layer7_0',
                'layer6_2_1': 'layer7_0',
                'layer7_0_0': 'layer7_0',
                'layer7_0_1': 'layer7_0',
                'decode_0': "decode",
                'decode_1': "decode",
                'decode_2': "decode",
                'decode_3': "decode",
                'decode_4': "decode",
                'decode_5': "decode"
            }

class MobileNetV2_512_Params_2_Groups(MobileNetV2_512_Params):
    def __init__(self):

        super().__init__()
        self.LAYER_GROUPS = \
            [
                [
                    "conv1",
                    "layer1_0_0",
                    "layer2_0_0",
                    "layer2_0_1",
                    "layer2_1_0",
                    "layer2_1_1",
                    "layer3_0_0",
                    "layer3_0_1",
                    "layer3_1_0",
                    "layer3_1_1",
                    "layer3_2_0",
                    "layer3_2_1",
                    "layer4_0_0",
                    "layer4_0_1",
                    "layer4_1_0",
                    "layer4_1_1",
                    "layer4_2_0",
                    "layer4_2_1",
                    "layer4_3_0",
                    "layer4_3_1",
                    "layer5_0_0",
                    "layer5_0_1",
                    "layer5_1_0",
                    "layer5_1_1",
                    "layer5_2_0",
                    "layer5_2_1",
                    "layer6_0_0",
                    "layer6_0_1",
                    "layer6_1_0",
                    "layer6_1_1",
                    "layer6_2_0",
                    "layer6_2_1",
                    "layer7_0_0",
                    "layer7_0_1",
                ],
                [
                    "decode_1",
                    "decode_2",
                    "decode_3",
                    "decode_4",
                    "decode_5",
                ]
            ]
        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1':      'layer7_0',
                'layer1_0_0': 'layer7_0',
                'layer2_0_0': 'layer7_0',
                'layer2_0_1': 'layer7_0',
                'layer2_1_0': 'layer7_0',
                'layer2_1_1': 'layer7_0',
                'layer3_0_0': 'layer7_0',
                'layer3_0_1': 'layer7_0',
                'layer3_1_0': 'layer7_0',
                'layer3_1_1': 'layer7_0',
                'layer3_2_0': 'layer7_0',
                'layer3_2_1': 'layer7_0',
                'layer4_0_0': 'layer7_0',
                'layer4_0_1': 'layer7_0',
                'layer4_1_0': 'layer7_0',
                'layer4_1_1': 'layer7_0',
                'layer4_2_0': 'layer7_0',
                'layer4_2_1': 'layer7_0',
                'layer4_3_0': 'layer7_0',
                'layer4_3_1': 'layer7_0',
                'layer5_0_0': 'layer7_0',
                'layer5_0_1': 'layer7_0',
                'layer5_1_0': 'layer7_0',
                'layer5_1_1': 'layer7_0',
                'layer5_2_0': 'layer7_0',
                'layer5_2_1': 'layer7_0',
                'layer6_0_0': 'layer7_0',
                'layer6_0_1': 'layer7_0',
                'layer6_1_0': 'layer7_0',
                'layer6_1_1': 'layer7_0',
                'layer6_2_0': 'layer7_0',
                'layer6_2_1': 'layer7_0',
                'layer7_0_0': 'layer7_0',
                'layer7_0_1': 'layer7_0',
                'decode_1': "decode",
                'decode_2': "decode",
                'decode_3': "decode",
                'decode_4': "decode",
                'decode_5': "decode"
            }

class MobileNetV2_256_Params_1_Groups(MobileNetV2_256_Params):
    def __init__(self):

        super().__init__()

        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1':      'decode',
                'layer1_0_0': 'decode',
                'layer2_0_0': 'decode',
                'layer2_0_1': 'decode',
                'layer2_1_0': 'decode',
                'layer2_1_1': 'decode',
                'layer3_0_0': 'decode',
                'layer3_0_1': 'decode',
                'layer3_1_0': 'decode',
                'layer3_1_1': 'decode',
                'layer3_2_0': 'decode',
                'layer3_2_1': 'decode',
                'layer4_0_0': 'decode',
                'layer4_0_1': 'decode',
                'layer4_1_0': 'decode',
                'layer4_1_1': 'decode',
                'layer4_2_0': 'decode',
                'layer4_2_1': 'decode',
                'layer4_3_0': 'decode',
                'layer4_3_1': 'decode',
                'layer5_0_0': 'decode',
                'layer5_0_1': 'decode',
                'layer5_1_0': 'decode',
                'layer5_1_1': 'decode',
                'layer5_2_0': 'decode',
                'layer5_2_1': 'decode',
                'layer6_0_0': 'decode',
                'layer6_0_1': 'decode',
                'layer6_1_0': 'decode',
                'layer6_1_1': 'decode',
                'layer6_2_0': 'decode',
                'layer6_2_1': 'decode',
                'layer7_0_0': 'decode',
                'layer7_0_1': 'decode',
                'decode_0': "decode",
                'decode_1': "decode",
                'decode_2': "decode",
                'decode_3': "decode",
                'decode_4': "decode",
                'decode_5': "decode"
            }

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'conv1': BLOCK_SIZES_256x256,
                'layer1_0_0': BLOCK_SIZES_256x256,
                'layer2_0_0': BLOCK_SIZES_256x256,
                'layer2_0_1': BLOCK_SIZES_256x256,
                'layer2_1_0': BLOCK_SIZES_256x256,
                'layer2_1_1': BLOCK_SIZES_256x256,
                'layer3_0_0': BLOCK_SIZES_256x256,
                'layer3_0_1': BLOCK_SIZES_256x256,
                'layer3_1_0': BLOCK_SIZES_256x256,
                'layer3_1_1': BLOCK_SIZES_256x256,
                'layer3_2_0': BLOCK_SIZES_256x256,
                'layer3_2_1': BLOCK_SIZES_256x256,
                'layer4_0_0': BLOCK_SIZES_256x256,
                'layer4_0_1': BLOCK_SIZES_256x256,
                'layer4_1_0': BLOCK_SIZES_256x256,
                'layer4_1_1': BLOCK_SIZES_256x256,
                'layer4_2_0': BLOCK_SIZES_256x256,
                'layer4_2_1': BLOCK_SIZES_256x256,
                'layer4_3_0': BLOCK_SIZES_256x256,
                'layer4_3_1': BLOCK_SIZES_256x256,
                'layer5_0_0': BLOCK_SIZES_256x256,
                'layer5_0_1': BLOCK_SIZES_256x256,
                'layer5_1_0': BLOCK_SIZES_256x256,
                'layer5_1_1': BLOCK_SIZES_256x256,
                'layer5_2_0': BLOCK_SIZES_256x256,
                'layer5_2_1': BLOCK_SIZES_256x256,
                'layer6_0_0': BLOCK_SIZES_256x256,
                'layer6_0_1': BLOCK_SIZES_256x256,
                'layer6_1_0': BLOCK_SIZES_256x256,
                'layer6_1_1': BLOCK_SIZES_256x256,
                'layer6_2_0': BLOCK_SIZES_256x256,
                'layer6_2_1': BLOCK_SIZES_256x256,
                'layer7_0_0': BLOCK_SIZES_256x256,
                'layer7_0_1': BLOCK_SIZES_256x256,
                'decode_0': BLOCK_SIZES_256x256,
                'decode_1': BLOCK_SIZES_256x256,
                'decode_2': BLOCK_SIZES_256x256,
                'decode_3': BLOCK_SIZES_256x256,
                'decode_4': BLOCK_SIZES_256x256,
                'decode_5': BLOCK_SIZES_256x256
            }

class MobileNetV2_256_Params_1_Groups_mini(MobileNetV2_256_Params_1_Groups):
    def __init__(self):

        super().__init__()

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'conv1': BLOCK_SIZES_MINI,
                'layer1_0_0': BLOCK_SIZES_MINI,
                'layer2_0_0': BLOCK_SIZES_MINI,
                'layer2_0_1': BLOCK_SIZES_MINI,
                'layer2_1_0': BLOCK_SIZES_MINI,
                'layer2_1_1': BLOCK_SIZES_MINI,
                'layer3_0_0': BLOCK_SIZES_MINI,
                'layer3_0_1': BLOCK_SIZES_MINI,
                'layer3_1_0': BLOCK_SIZES_MINI,
                'layer3_1_1': BLOCK_SIZES_MINI,
                'layer3_2_0': BLOCK_SIZES_MINI,
                'layer3_2_1': BLOCK_SIZES_MINI,
                'layer4_0_0': BLOCK_SIZES_MINI,
                'layer4_0_1': BLOCK_SIZES_MINI,
                'layer4_1_0': BLOCK_SIZES_MINI,
                'layer4_1_1': BLOCK_SIZES_MINI,
                'layer4_2_0': BLOCK_SIZES_MINI,
                'layer4_2_1': BLOCK_SIZES_MINI,
                'layer4_3_0': BLOCK_SIZES_MINI,
                'layer4_3_1': BLOCK_SIZES_MINI,
                'layer5_0_0': BLOCK_SIZES_MINI,
                'layer5_0_1': BLOCK_SIZES_MINI,
                'layer5_1_0': BLOCK_SIZES_MINI,
                'layer5_1_1': BLOCK_SIZES_MINI,
                'layer5_2_0': BLOCK_SIZES_MINI,
                'layer5_2_1': BLOCK_SIZES_MINI,
                'layer6_0_0': BLOCK_SIZES_MINI,
                'layer6_0_1': BLOCK_SIZES_MINI,
                'layer6_1_0': BLOCK_SIZES_MINI,
                'layer6_1_1': BLOCK_SIZES_MINI,
                'layer6_2_0': BLOCK_SIZES_MINI,
                'layer6_2_1': BLOCK_SIZES_MINI,
                'layer7_0_0': BLOCK_SIZES_MINI,
                'layer7_0_1': BLOCK_SIZES_MINI,
                'decode_0': BLOCK_SIZES_MINI,
                'decode_1': BLOCK_SIZES_MINI,
                'decode_2': BLOCK_SIZES_MINI,
                'decode_3': BLOCK_SIZES_MINI,
                'decode_4': BLOCK_SIZES_MINI,
                'decode_5': BLOCK_SIZES_MINI
            }

        self.LAYER_GROUPS = \
            [
                [
                    "conv1",
                    "layer1_0_0",
                    "layer2_0_0",
                    "layer2_0_1",
                    "layer2_1_0",
                    "layer2_1_1",
                    "layer3_0_0",
                    "layer3_0_1",
                    "layer3_1_0",
                    "layer3_1_1",
                    "layer3_2_0",
                    "layer3_2_1",
                    "layer4_0_0",
                    "layer4_0_1",
                    "layer4_1_0",
                    "layer4_1_1",
                    "layer4_2_0",
                    "layer4_2_1",
                    "layer4_3_0",
                    "layer4_3_1",
                    "layer5_0_0",
                    "layer5_0_1",
                    "layer5_1_0",
                    "layer5_1_1",
                    "layer5_2_0",
                    "layer5_2_1",
                    "layer6_0_0",
                    "layer6_0_1",
                    "layer6_1_0",
                    "layer6_1_1",
                    "layer6_2_0",
                    "layer6_2_1",
                    "layer7_0_0",
                    "layer7_0_1",
                    "decode_0",
                    "decode_1",
                    "decode_2",
                    "decode_3",
                    "decode_4",
                    "decode_5",
                ]
            ]

        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1':      'decode',
                'layer1_0_0': 'decode',
                'layer2_0_0': 'decode',
                'layer2_0_1': 'decode',
                'layer2_1_0': 'decode',
                'layer2_1_1': 'decode',
                'layer3_0_0': 'decode',
                'layer3_0_1': 'decode',
                'layer3_1_0': 'decode',
                'layer3_1_1': 'decode',
                'layer3_2_0': 'decode',
                'layer3_2_1': 'decode',
                'layer4_0_0': 'decode',
                'layer4_0_1': 'decode',
                'layer4_1_0': 'decode',
                'layer4_1_1': 'decode',
                'layer4_2_0': 'decode',
                'layer4_2_1': 'decode',
                'layer4_3_0': 'decode',
                'layer4_3_1': 'decode',
                'layer5_0_0': 'decode',
                'layer5_0_1': 'decode',
                'layer5_1_0': 'decode',
                'layer5_1_1': 'decode',
                'layer5_2_0': 'decode',
                'layer5_2_1': 'decode',
                'layer6_0_0': 'decode',
                'layer6_0_1': 'decode',
                'layer6_1_0': 'decode',
                'layer6_1_1': 'decode',
                'layer6_2_0': 'decode',
                'layer6_2_1': 'decode',
                'layer7_0_0': 'decode',
                'layer7_0_1': 'decode',
                'decode_0': "decode",
                'decode_1': "decode",
                'decode_2': "decode",
                'decode_3': "decode",
                'decode_4': "decode",
                'decode_5': "decode"
            }
