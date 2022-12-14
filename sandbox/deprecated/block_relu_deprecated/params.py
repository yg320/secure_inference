import json

BLOCK_SIZES_FULL = \
    [[1, 1],
     [1, 2],
     [2, 1],
     [1, 3],
     [3, 1],
     [1, 4],
     [2, 2],
     [4, 1],
     [1, 5],
     [5, 1],
     [1, 6],
     [2, 3],
     [3, 2],
     [6, 1],
     # [1, 7],
     # [7, 1],
     # [1, 8],
     [2, 4],
     [4, 2],
     # [8, 1],
     # [1, 9],
     [3, 3],
     # [9, 1],
     # [1, 10],
     [2, 5],
     [5, 2],
     # [10, 1],
     # [1, 11],
     # [11, 1],
     # [1, 12],
     [2, 6],
     [3, 4],
     [4, 3],
     [6, 2],
     # [12, 1],
     # [1, 13],
     # [13, 1],
     # [1, 14],
     [2, 7],
     [7, 2],
     # [14, 1],
     # [1, 15],
     [3, 5],
     [5, 3],
     # [15, 1],
     # [1, 16],
     [2, 8],
     [4, 4],
     [8, 2],
     # [16, 1],
     # [2, 9],
     [3, 6],
     [6, 3],
     # [9, 2],
     [2, 10],
     [4, 5],
     [5, 4],
     [10, 2],
     [3, 7],
     [7, 3],
     # [2, 11],
     # [11, 2],
     [2, 12],
     [3, 8],
     [4, 6],
     [6, 4],
     [8, 3],
     [12, 2],
     [5, 5],
     # [2, 13],
     # [13, 2],
     [3, 9],
     [9, 3],
     # [2, 14],
     [4, 7],
     [7, 4],
     # [14, 2],
     # [2, 15],
     # [3, 10],
     [5, 6],
     [6, 5],
     # [10, 3],
     # [15, 2],
     [2, 16],
     [4, 8],
     [8, 4],
     [16, 2],
     # [3, 11],
     # [11, 3],
     # [5, 7],
     # [7, 5],
     [3, 12],
     # [4, 9],
     [6, 6],
     # [9, 4],
     [12, 3],
     # [3, 13],
     # [13, 3],
     [4, 10],
     [5, 8],
     [8, 5],
     [10, 4],
     # [3, 14],
     # [6, 7],
     # [7, 6],
     # [14, 3],
     # [4, 11],
     # [11, 4],
     [3, 15],
     # [5, 9],
     # [9, 5],
     [15, 3],
     # [3, 16],
     [4, 12],
     [6, 8],
     [8, 6],
     [12, 4],
     # [16, 3],
     [7, 7],
     [5, 10],
     [10, 5],
     # [4, 13],
     # [13, 4],
     [6, 9],
     [9, 6],
     # [5, 11],
     # [11, 5],
     # [4, 14],
     # [7, 8],
     # [8, 7],
     # [14, 4],
     # [4, 15],
     [5, 12],
     [6, 10],
     [10, 6],
     [12, 5],
     # [15, 4],
     # [7, 9],
     # [9, 7],
     [4, 16],
     [8, 8],
     [16, 4],
     # [5, 13],
     # [13, 5],
     # [6, 11],
     # [11, 6],
     # [5, 14],
     # [7, 10],
     # [10, 7],
     # [14, 5],
     [6, 12],
     # [8, 9],
     # [9, 8],
     [12, 6],
     [5, 15],
     [15, 5],
     # [7, 11],
     # [11, 7],
     # [6, 13],
     # [13, 6],
     # [5, 16],
     [8, 10],
     [10, 8],
     # [16, 5],
     [9, 9],
     # [6, 14],
     # [7, 12],
     # [12, 7],
     # [14, 6],
     # [8, 11],
     # [11, 8],
     # [6, 15],
     # [9, 10],
     # [10, 9],
     # [15, 6],
     # [7, 13],
     # [13, 7],
     [6, 16],
     [8, 12],
     [12, 8],
     [16, 6],
     # [7, 14],
     # [14, 7],
     # [9, 11],
     # [11, 9],
     [10, 10],
     # [8, 13],
     # [13, 8],
     # [7, 15],
     # [15, 7],
     [9, 12],
     [12, 9],
     # [10, 11],
     # [11, 10],
     # [7, 16],
     [8, 14],
     [14, 8],
     # [16, 7],
     # [9, 13],
     # [13, 9],
     # [8, 15],
     # [10, 12],
     # [12, 10],
     # [15, 8],
     [11, 11],
     # [9, 14],
     # [14, 9],
     [8, 16],
     [16, 8],
     # [10, 13],
     # [13, 10],
     # [11, 12],
     # [12, 11],
     # [9, 15],
     # [15, 9],
     [10, 14],
     [14, 10],
     # [11, 13],
     # [13, 11],
     # [9, 16],
     [12, 12],
     # [16, 9],
     [10, 15],
     [15, 10],
     # [11, 14],
     # [14, 11],
     # [12, 13],
     # [13, 12],
     [10, 16],
     [16, 10],
     # [11, 15],
     # [15, 11],
     [12, 14],
     [14, 12],
     [13, 13],
     # [11, 16],
     # [16, 11],
     # [12, 15],
     # [15, 12],
     # [13, 14],
     # [14, 13],
     [12, 16],
     [16, 12],
     # [13, 15],
     # [15, 13],
     [14, 14],
     # [13, 16],
     # [16, 13],
     # [14, 15],
     # [15, 14],
     # [14, 16],
     # [16, 14],
     [15, 15],
     # [15, 16],
     # [16, 15],
     [16, 16],
     [32, 32],
     [64, 64]
     ]

BLOCK_SIZES_MINI = [
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
    [2, 4],
    [4, 2],
    [3, 3],
    [4, 4],
    [3, 6],
    [6, 3],
    [5, 5],
    [4, 8],
    [8, 4],
    [6, 6],
    [7, 7],
    [5, 10],
    [10, 5],
    [8, 8],
    [6, 12],
    [12, 6],
    [9, 9],
    [7, 14],
    [14, 7],
    [10, 10],
    [11, 11],
    [8, 16],
    [16, 8],
    [12, 12],
    [13, 13],
    [14, 14],
    [15, 15],
    [16, 16],
    [64, 64]
]

BLOCK_SIZE_COMPLETE = [[i, j] for i in range(1, 65) for j in range(1, 65)]


class Params:
    def __init__(self, BLOCK_NAMES, LAYER_NAME_TO_BLOCK_NAME, LAYER_NAMES, IN_LAYER_PROXY_SPEC, LAYER_NAME_TO_CHANNELS,
                 LAYER_NAME_TO_BLOCK_SIZES, LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS,
                 LAYER_NAME_TO_LAYER_DIM, LAYER_NAME_TO_RELU_COUNT, LAYER_HIERARCHY_SPEC):
        self.BLOCK_NAMES = BLOCK_NAMES
        self.LAYER_NAME_TO_BLOCK_NAME = LAYER_NAME_TO_BLOCK_NAME
        self.LAYER_NAMES = LAYER_NAMES
        self.IN_LAYER_PROXY_SPEC = IN_LAYER_PROXY_SPEC
        self.LAYER_NAME_TO_CHANNELS = LAYER_NAME_TO_CHANNELS
        self.LAYER_NAME_TO_BLOCK_SIZES = LAYER_NAME_TO_BLOCK_SIZES
        self.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS = LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS
        self.LAYER_NAME_TO_LAYER_DIM = LAYER_NAME_TO_LAYER_DIM
        self.LAYER_NAME_TO_RELU_COUNT = LAYER_NAME_TO_RELU_COUNT
        self.LAYER_HIERARCHY_SPEC = LAYER_HIERARCHY_SPEC


class MobileNetV2_256_Params:
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
                "conv1",  # self.conv1.activate

                "layer1_0_0",  # layer1[0].conv[0].activate
                # "layer1_0_1",

                "layer2_0_0",  # layer2[0].conv[0].activate
                "layer2_0_1",  # layer2[0].conv[1].activate
                # "layer2_0_2",

                "layer2_1_0",  # layer2[1].conv[0].activate
                "layer2_1_1",  # layer2[1].conv[1].activate
                # "layer2_1_2",

                "layer3_0_0",  # layer3[0].conv[0].activate
                "layer3_0_1",  # layer3[0].conv[1].activate
                # "layer3_0_2",

                "layer3_1_0",  # layer3[1].conv[0].activate
                "layer3_1_1",  # layer3[1].conv[1].activate
                # "layer3_1_2",

                "layer3_2_0",  # layer3[2].conv[0].activate
                "layer3_2_1",  # layer3[2].conv[1].activate
                # "layer3_2_2",

                "layer4_0_0",  # layer4[0].conv[0].activate
                "layer4_0_1",  # layer4[0].conv[1].activate
                # "layer4_0_2",

                "layer4_1_0",  # layer4[1].conv[0].activate
                "layer4_1_1",  # layer4[1].conv[1].activate
                # "layer4_1_2",

                "layer4_2_0",  # layer4[2].conv[0].activate
                "layer4_2_1",  # layer4[2].conv[1].activate
                # "layer4_2_2",

                "layer4_3_0",  # layer4[3].conv[0].activate
                "layer4_3_1",  # layer4[3].conv[1].activate
                # "layer4_3_2",

                "layer5_0_0",  # layer5[0].conv[0].activate
                "layer5_0_1",  # layer5[0].conv[1].activate
                # "layer5_0_2",

                "layer5_1_0",  # layer5[1].conv[0].activate
                "layer5_1_1",  # layer5[1].conv[1].activate
                # "layer5_1_2",

                "layer5_2_0",  # layer5[2].conv[0].activate
                "layer5_2_1",  # layer5[2].conv[1].activate
                # "layer5_2_2",

                "layer6_0_0",  # layer6[0].conv[0].activate
                "layer6_0_1",  # layer6[0].conv[1].activate
                # "layer6_0_2",

                "layer6_1_0",  # layer6[1].conv[0].activate
                "layer6_1_1",  # layer6[1].conv[1].activate
                # "layer6_1_2",

                "layer6_2_0",  # layer6[2].conv[0].activate
                "layer6_2_1",  # layer6[2].conv[1].activate
                # "layer6_2_2",

                "layer7_0_0",  # layer7[0].conv[0].activate
                "layer7_0_1",  # layer7[0].conv[1].activate
                # "layer7_0_2",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]

        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1': 'layer4_0',
                'layer1_0_0': 'layer4_0',
                'layer2_0_0': 'layer4_0',
                'layer2_0_1': 'layer4_0',
                'layer2_1_0': 'layer4_0',
                'layer2_1_1': 'layer4_0',
                'layer3_0_0': 'layer4_0',
                'layer3_0_1': 'layer4_0',
                'layer3_1_0': 'layer4_0',
                'layer3_1_1': 'layer4_0',
                'layer3_2_0': 'layer4_0',
                'layer3_2_1': 'layer4_0',
                'layer4_0_0': 'layer6_0',
                'layer4_0_1': 'layer6_0',
                'layer4_1_0': 'layer6_0',
                'layer4_1_1': 'layer6_0',
                'layer4_2_0': 'layer6_0',
                'layer4_2_1': 'layer6_0',
                'layer4_3_0': 'layer6_0',
                'layer4_3_1': 'layer6_0',
                'layer5_0_0': 'layer6_0',
                'layer5_0_1': 'layer6_0',
                'layer5_1_0': 'layer6_0',
                'layer5_1_1': 'layer6_0',
                'layer5_2_0': 'layer6_0',
                'layer5_2_1': 'layer6_0',
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

        self.LAYER_NAME_TO_DIMS = {'conv1': [32, 256, 256], 'layer1_0_0': [32, 128, 128], 'layer2_0_0': [96, 128, 128],
                                   'layer2_0_1': [96, 64, 64], 'layer2_1_0': [144, 64, 64], 'layer2_1_1': [144, 64, 64],
                                   'layer3_0_0': [144, 64, 64], 'layer3_0_1': [144, 32, 32],
                                   'layer3_1_0': [192, 32, 32], 'layer3_1_1': [192, 32, 32],
                                   'layer3_2_0': [192, 32, 32], 'layer3_2_1': [192, 32, 32],
                                   'layer4_0_0': [192, 32, 32], 'layer4_0_1': [192, 32, 32],
                                   'layer4_1_0': [384, 32, 32], 'layer4_1_1': [384, 32, 32],
                                   'layer4_2_0': [384, 32, 32], 'layer4_2_1': [384, 32, 32],
                                   'layer4_3_0': [384, 32, 32], 'layer4_3_1': [384, 32, 32],
                                   'layer5_0_0': [384, 32, 32], 'layer5_0_1': [384, 32, 32],
                                   'layer5_1_0': [576, 32, 32], 'layer5_1_1': [576, 32, 32],
                                   'layer5_2_0': [576, 32, 32], 'layer5_2_1': [576, 32, 32],
                                   'layer6_0_0': [576, 32, 32], 'layer6_0_1': [576, 32, 32],
                                   'layer6_1_0': [960, 32, 32], 'layer6_1_1': [960, 32, 32],
                                   'layer6_2_0': [960, 32, 32], 'layer6_2_1': [960, 32, 32],
                                   'layer7_0_0': [960, 32, 32], 'layer7_0_1': [960, 32, 32], 'decode_0': [512, 1, 1],
                                   'decode_1': [512, 32, 32], 'decode_2': [512, 32, 32], 'decode_3': [512, 32, 32],
                                   'decode_4': [512, 32, 32], 'decode_5': [512, 32, 32]}


class MobileNetV2Params:
    def __init__(self):
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
        self.BLOCK_NAMES_TO_BLOCK_INDEX = \
            {
                "conv1": 0,
                "layer1_0": 1,
                "layer2_0": 2,
                "layer2_1": 3,
                "layer3_0": 4,
                "layer3_1": 5,
                "layer3_2": 6,
                "layer4_0": 7,
                "layer4_1": 8,
                "layer4_2": 9,
                "layer4_3": 10,
                "layer5_0": 11,
                "layer5_1": 12,
                "layer5_2": 13,
                "layer6_0": 14,
                "layer6_1": 15,
                "layer6_2": 16,
                "layer7_0": 17,
                "decode": 18,
                None: 19,
            }

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
                "conv1",  # self.conv1.activate

                "layer1_0_0",  # layer1[0].conv[0].activate
                # "layer1_0_1",

                "layer2_0_0",  # layer2[0].conv[0].activate
                "layer2_0_1",  # layer2[0].conv[1].activate
                # "layer2_0_2",

                "layer2_1_0",  # layer2[1].conv[0].activate
                "layer2_1_1",  # layer2[1].conv[1].activate
                # "layer2_1_2",

                "layer3_0_0",  # layer3[0].conv[0].activate
                "layer3_0_1",  # layer3[0].conv[1].activate
                # "layer3_0_2",

                "layer3_1_0",  # layer3[1].conv[0].activate
                "layer3_1_1",  # layer3[1].conv[1].activate
                # "layer3_1_2",

                "layer3_2_0",  # layer3[2].conv[0].activate
                "layer3_2_1",  # layer3[2].conv[1].activate
                # "layer3_2_2",

                "layer4_0_0",  # layer4[0].conv[0].activate
                "layer4_0_1",  # layer4[0].conv[1].activate
                # "layer4_0_2",

                "layer4_1_0",  # layer4[1].conv[0].activate
                "layer4_1_1",  # layer4[1].conv[1].activate
                # "layer4_1_2",

                "layer4_2_0",  # layer4[2].conv[0].activate
                "layer4_2_1",  # layer4[2].conv[1].activate
                # "layer4_2_2",

                "layer4_3_0",  # layer4[3].conv[0].activate
                "layer4_3_1",  # layer4[3].conv[1].activate
                # "layer4_3_2",

                "layer5_0_0",  # layer5[0].conv[0].activate
                "layer5_0_1",  # layer5[0].conv[1].activate
                # "layer5_0_2",

                "layer5_1_0",  # layer5[1].conv[0].activate
                "layer5_1_1",  # layer5[1].conv[1].activate
                # "layer5_1_2",

                "layer5_2_0",  # layer5[2].conv[0].activate
                "layer5_2_1",  # layer5[2].conv[1].activate
                # "layer5_2_2",

                "layer6_0_0",  # layer6[0].conv[0].activate
                "layer6_0_1",  # layer6[0].conv[1].activate
                # "layer6_0_2",

                "layer6_1_0",  # layer6[1].conv[0].activate
                "layer6_1_1",  # layer6[1].conv[1].activate
                # "layer6_1_2",

                "layer6_2_0",  # layer6[2].conv[0].activate
                "layer6_2_1",  # layer6[2].conv[1].activate
                # "layer6_2_2",

                "layer7_0_0",  # layer7[0].conv[0].activate
                "layer7_0_1",  # layer7[0].conv[1].activate
                # "layer7_0_2",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]

        self.IN_LAYER_PROXY_SPEC = \
            {
                'conv1': 'layer2_1',
                'layer1_0_0': 'layer3_0',
                'layer2_0_0': 'layer3_1',
                'layer2_0_1': 'layer3_1',
                'layer2_1_0': 'layer3_2',
                'layer2_1_1': 'layer3_2',
                'layer3_0_0': 'layer4_0',
                'layer3_0_1': 'layer4_0',
                'layer3_1_0': 'layer4_1',
                'layer3_1_1': 'layer4_1',
                'layer3_2_0': 'layer4_2',
                'layer3_2_1': 'layer4_2',
                'layer4_0_0': 'layer4_3',
                'layer4_0_1': 'layer4_3',
                'layer4_1_0': 'layer5_0',
                'layer4_1_1': 'layer5_0',
                'layer4_2_0': 'layer5_1',
                'layer4_2_1': 'layer5_1',
                'layer4_3_0': 'layer5_2',
                'layer4_3_1': 'layer5_2',
                'layer5_0_0': 'layer6_0',
                'layer5_0_1': 'layer6_0',
                'layer5_1_0': 'layer6_1',
                'layer5_1_1': 'layer6_1',
                'layer5_2_0': 'layer6_2',
                'layer5_2_1': 'layer6_2',
                'layer6_0_0': 'decode',
                'layer6_0_1': 'decode',
                'layer6_1_0': 'decode',
                'layer6_1_1': 'decode',
                'layer6_2_0': 'decode',
                'layer6_2_1': 'decode',
                'layer7_0_0': 'decode',
                'layer7_0_1': 'decode',
                'decode_0': None,
                'decode_1': None,
                'decode_2': None,
                'decode_3': None,
                'decode_4': None,
                'decode_5': None
            }

        self.LAYER_NAME_TO_LAYER_SIZE = {'conv1': 256, 'layer1_0_0': 256, 'layer2_0_0': 256, 'layer2_0_1': 128,
                                         'layer2_1_0': 128, 'layer2_1_1': 128, 'layer3_0_0': 128, 'layer3_0_1': 64,
                                         'layer3_1_0': 64, 'layer3_1_1': 64, 'layer3_2_0': 64, 'layer3_2_1': 64,
                                         'layer4_0_0': 64, 'layer4_0_1': 64, 'layer4_1_0': 64, 'layer4_1_1': 64,
                                         'layer4_2_0': 64, 'layer4_2_1': 64, 'layer4_3_0': 64, 'layer4_3_1': 64,
                                         'layer5_0_0': 64, 'layer5_0_1': 64, 'layer5_1_0': 64, 'layer5_1_1': 64,
                                         'layer5_2_0': 64, 'layer5_2_1': 64, 'layer6_0_0': 64, 'layer6_0_1': 64,
                                         'layer6_1_0': 64, 'layer6_1_1': 64, 'layer6_2_0': 64, 'layer6_2_1': 64,
                                         'layer7_0_0': 64, 'layer7_0_1': 64, 'decode_0': 1, 'decode_1': 64,
                                         'decode_2': 64, 'decode_3': 64, 'decode_4': 64, 'decode_5': 64}

        self.LAYER_NAME_TO_CHANNELS = \
            {'conv1': 32, 'layer1_0_0': 32, 'layer2_0_0': 96, 'layer2_0_1': 96, 'layer2_1_0': 144, 'layer2_1_1': 144,
             'layer3_0_0': 144, 'layer3_0_1': 144, 'layer3_1_0': 192, 'layer3_1_1': 192, 'layer3_2_0': 192,
             'layer3_2_1': 192, 'layer4_0_0': 192, 'layer4_0_1': 192, 'layer4_1_0': 384, 'layer4_1_1': 384,
             'layer4_2_0': 384, 'layer4_2_1': 384, 'layer4_3_0': 384, 'layer4_3_1': 384, 'layer5_0_0': 384,
             'layer5_0_1': 384, 'layer5_1_0': 576, 'layer5_1_1': 576, 'layer5_2_0': 576, 'layer5_2_1': 576,
             'layer6_0_0': 576, 'layer6_0_1': 576, 'layer6_1_0': 960, 'layer6_1_1': 960, 'layer6_2_0': 960,
             'layer6_2_1': 960, 'layer7_0_0': 960, 'layer7_0_1': 960, 'decode_0': 512, 'decode_1': 512, 'decode_2': 512,
             'decode_3': 512, 'decode_4': 512, 'decode_5': 512}

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

        self.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS = {
            'conv1': [32, 16, 8, 4, 2, 1],
            'layer1_0_0': [32, 16, 8, 4, 2, 1],
            'layer2_0_0': [48, 16, 8, 4, 2, 1],
            'layer2_0_1': [48, 16, 8, 4, 2, 1],
            'layer2_1_0': [48, 16, 8, 4, 2, 1],
            'layer2_1_1': [48, 16, 8, 4, 2, 1],
            'layer3_0_0': [48, 16, 8, 4, 2, 1],
            'layer3_0_1': [48, 16, 8, 4, 2, 1],
            'layer3_1_0': [48, 16, 8, 4, 2, 1],
            'layer3_1_1': [48, 16, 8, 4, 2, 1],
            'layer3_2_0': [48, 16, 8, 4, 2, 1],
            'layer3_2_1': [48, 16, 8, 4, 2, 1],
            'layer4_0_0': [48, 16, 8, 4, 2, 1],
            'layer4_0_1': [48, 16, 8, 4, 2, 1],
            'layer4_1_0': [48, 16, 8, 4, 2, 1],
            'layer4_1_1': [48, 16, 8, 4, 2, 1],
            'layer4_2_0': [48, 16, 8, 4, 2, 1],
            'layer4_2_1': [48, 16, 8, 4, 2, 1],
            'layer4_3_0': [48, 16, 8, 4, 2, 1],
            'layer4_3_1': [48, 16, 8, 4, 2, 1],
            'layer5_0_0': [48, 16, 8, 4, 2, 1],
            'layer5_0_1': [48, 16, 8, 4, 2, 1],
            'layer5_1_0': [48, 16, 8, 4, 2, 1],
            'layer5_1_1': [48, 16, 8, 4, 2, 1],
            'layer5_2_0': [48, 16, 8, 4, 2, 1],
            'layer5_2_1': [48, 16, 8, 4, 2, 1],
            'layer6_0_0': [48, 16, 8, 4, 2, 1],
            'layer6_0_1': [48, 16, 8, 4, 2, 1],
            'layer6_1_0': [48, 16, 8, 4, 2, 1],
            'layer6_1_1': [48, 16, 8, 4, 2, 1],
            'layer6_2_0': [48, 16, 8, 4, 2, 1],
            'layer6_2_1': [48, 16, 8, 4, 2, 1],
            'layer7_0_0': [48, 16, 8, 4, 2, 1],
            'layer7_0_1': [48, 16, 8, 4, 2, 1],
            'decode_0': [32, 16, 8, 4, 2, 1],
            'decode_1': [32, 16, 8, 4, 2, 1],
            'decode_2': [32, 16, 8, 4, 2, 1],
            'decode_3': [32, 16, 8, 4, 2, 1],
            'decode_4': [32, 16, 8, 4, 2, 1],
            'decode_5': [32, 16, 8, 4, 2, 1],
        }

        self.LAYER_NAME_TO_RELU_COUNT = {
            'conv1': 2097152, 'layer1_0_0': 2097152, 'layer2_0_0': 6291456,
            'layer2_0_1': 1572864, 'layer2_1_0': 2359296, 'layer2_1_1': 2359296,
            'layer3_0_0': 2359296, 'layer3_0_1': 589824, 'layer3_1_0': 786432,
            'layer3_1_1': 786432, 'layer3_2_0': 786432, 'layer3_2_1': 786432,
            'layer4_0_0': 786432, 'layer4_0_1': 786432, 'layer4_1_0': 1572864,
            'layer4_1_1': 1572864, 'layer4_2_0': 1572864, 'layer4_2_1': 1572864,
            'layer4_3_0': 1572864, 'layer4_3_1': 1572864, 'layer5_0_0': 1572864,
            'layer5_0_1': 1572864, 'layer5_1_0': 2359296, 'layer5_1_1': 2359296,
            'layer5_2_0': 2359296, 'layer5_2_1': 2359296, 'layer6_0_0': 2359296,
            'layer6_0_1': 2359296, 'layer6_1_0': 3932160, 'layer6_1_1': 3932160,
            'layer6_2_0': 3932160, 'layer6_2_1': 3932160, 'layer7_0_0': 3932160,
            'layer7_0_1': 3932160, 'decode_0': 512, 'decode_1': 2097152,
            'decode_2': 2097152, 'decode_3': 2097152, 'decode_4': 2097152,
            'decode_5': 2097152
        }

        self.LAYER_HIERARCHY_SPEC = [
            # [['conv1'], ['layer1_0_0'], ['layer2_0_0'], ['layer2_0_1'], ['layer2_1_0'], ['layer2_1_1'], ['layer3_0_0'],
            #  ['layer3_0_1'], ['layer3_1_0'], ['layer3_1_1'], ['layer3_2_0'], ['layer3_2_1'], ['layer4_0_0'],
            #  ['layer4_0_1'], ['layer4_1_0'], ['layer4_1_1'], ['layer4_2_0'], ['layer4_2_1'], ['layer4_3_0'],
            #  ['layer4_3_1'], ['layer5_0_0'], ['layer5_0_1'], ['layer5_1_0'], ['layer5_1_1'], ['layer5_2_0'],
            #  ['layer5_2_1'], ['layer6_0_0'], ['layer6_0_1'], ['layer6_1_0'], ['layer6_1_1'], ['layer6_2_0'],
            #  ['layer6_2_1'], ['layer7_0_0'], ['layer7_0_1'], ['decode_0'], ['decode_1'], ['decode_2'], ['decode_3'],
            #  ['decode_4'], ['decode_5']]
            #

            [['conv1'],
             ['layer1_0_0'],
             ['layer2_0_0', 'layer2_0_1'],
             ['layer2_1_0', 'layer2_1_1'],
             ['layer3_0_0', 'layer3_0_1'],
             ['layer3_1_0', 'layer3_1_1'],
             ['layer3_2_0', 'layer3_2_1'],
             ['layer4_0_0', 'layer4_0_1'],
             ['layer4_1_0', 'layer4_1_1'],
             ['layer4_2_0', 'layer4_2_1'],
             ['layer4_3_0', 'layer4_3_1'],
             ['layer5_0_0', 'layer5_0_1'],
             ['layer5_1_0', 'layer5_1_1'],
             ['layer5_2_0', 'layer5_2_1'],
             ['layer6_0_0', 'layer6_0_1'],
             ['layer6_1_0', 'layer6_1_1'],
             ['layer6_2_0', 'layer6_2_1'],
             ['layer7_0_0', 'layer7_0_1'],
             ['decode_0', 'decode_1', 'decode_2', 'decode_3', 'decode_4', 'decode_5']],

            [['conv1', 'layer1_0_0'],
             ['layer2_0_0', 'layer2_0_1', 'layer2_1_0', 'layer2_1_1'],
             ['layer3_0_0', 'layer3_0_1', 'layer3_1_0', 'layer3_1_1', 'layer3_2_0', 'layer3_2_1'],
             ['layer4_0_0', 'layer4_0_1', 'layer4_1_0', 'layer4_1_1', 'layer4_2_0', 'layer4_2_1', 'layer4_3_0',
              'layer4_3_1'],
             ['layer5_0_0', 'layer5_0_1', 'layer5_1_0', 'layer5_1_1', 'layer5_2_0', 'layer5_2_1'],
             ['layer6_0_0', 'layer6_0_1', 'layer6_1_0', 'layer6_1_1', 'layer6_2_0', 'layer6_2_1'],
             ['layer7_0_0', 'layer7_0_1'],
             ['decode_0', 'decode_1', 'decode_2', 'decode_3', 'decode_4', 'decode_5']],

            [self.LAYER_NAMES]

        ]


class ResNetParams:
    def __init__(self, HIERARCHY_NAME, LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS, LAYER_HIERARCHY_SPEC,
                 DATASET, CONFIG, CHECKPOINT):
        self.BACKBONE = "ResNetV1c"
        self.DATASET = DATASET
        self.CONFIG = CONFIG
        self.CHECKPOINT = CHECKPOINT

        self.BLOCK_NAMES = [
            'stem',
            'layer1_0',
            'layer1_1',
            'layer1_2',
            'layer2_0',
            'layer2_1',
            'layer2_2',
            'layer2_3',
            'layer3_0',
            'layer3_1',
            'layer3_2',
            'layer3_3',
            'layer3_4',
            'layer3_5',
            'layer4_0',
            'layer4_1',
            'layer4_2',
            'decode',
            # 'cls_decode',
            None
        ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                "stem_2": "stem",
                "stem_5": "stem",
                "stem_8": "stem",
                "layer1_0_1": "layer1_0",
                "layer1_0_2": "layer1_0",
                "layer1_0_3": "layer1_0",
                "layer1_1_1": "layer1_1",
                "layer1_1_2": "layer1_1",
                "layer1_1_3": "layer1_1",
                "layer1_2_1": "layer1_2",
                "layer1_2_2": "layer1_2",
                "layer1_2_3": "layer1_2",
                "layer2_0_1": "layer2_0",
                "layer2_0_2": "layer2_0",
                "layer2_0_3": "layer2_0",
                "layer2_1_1": "layer2_1",
                "layer2_1_2": "layer2_1",
                "layer2_1_3": "layer2_1",
                "layer2_2_1": "layer2_2",
                "layer2_2_2": "layer2_2",
                "layer2_2_3": "layer2_2",
                "layer2_3_1": "layer2_3",
                "layer2_3_2": "layer2_3",
                "layer2_3_3": "layer2_3",
                "layer3_0_1": "layer3_0",
                "layer3_0_2": "layer3_0",
                "layer3_0_3": "layer3_0",
                "layer3_1_1": "layer3_1",
                "layer3_1_2": "layer3_1",
                "layer3_1_3": "layer3_1",
                "layer3_2_1": "layer3_2",
                "layer3_2_2": "layer3_2",
                "layer3_2_3": "layer3_2",
                "layer3_3_1": "layer3_3",
                "layer3_3_2": "layer3_3",
                "layer3_3_3": "layer3_3",
                "layer3_4_1": "layer3_4",
                "layer3_4_2": "layer3_4",
                "layer3_4_3": "layer3_4",
                "layer3_5_1": "layer3_5",
                "layer3_5_2": "layer3_5",
                "layer3_5_3": "layer3_5",
                "layer4_0_1": "layer4_0",
                "layer4_0_2": "layer4_0",
                "layer4_0_3": "layer4_0",
                "layer4_1_1": "layer4_1",
                "layer4_1_2": "layer4_1",
                "layer4_1_3": "layer4_1",
                "layer4_2_1": "layer4_2",
                "layer4_2_2": "layer4_2",
                "layer4_2_3": "layer4_2",
                "decode_0": "decode",
                "decode_1": "decode",
                "decode_2": "decode",
                "decode_3": "decode",
                "decode_4": "decode",
                "decode_5": "decode",
            }

        self.LAYER_NAMES = \
            [
                "stem_2",
                "stem_5",
                "stem_8",
                "layer1_0_1",
                "layer1_0_2",
                "layer1_0_3",
                "layer1_1_1",
                "layer1_1_2",
                "layer1_1_3",
                "layer1_2_1",
                "layer1_2_2",
                "layer1_2_3",
                "layer2_0_1",
                "layer2_0_2",
                "layer2_0_3",
                "layer2_1_1",
                "layer2_1_2",
                "layer2_1_3",
                "layer2_2_1",
                "layer2_2_2",
                "layer2_2_3",
                "layer2_3_1",
                "layer2_3_2",
                "layer2_3_3",
                "layer3_0_1",
                "layer3_0_2",
                "layer3_0_3",
                "layer3_1_1",
                "layer3_1_2",
                "layer3_1_3",
                "layer3_2_1",
                "layer3_2_2",
                "layer3_2_3",
                "layer3_3_1",
                "layer3_3_2",
                "layer3_3_3",
                "layer3_4_1",
                "layer3_4_2",
                "layer3_4_3",
                "layer3_5_1",
                "layer3_5_2",
                "layer3_5_3",
                "layer4_0_1",
                "layer4_0_2",
                "layer4_0_3",
                "layer4_1_1",
                "layer4_1_2",
                "layer4_1_3",
                "layer4_2_1",
                "layer4_2_2",
                "layer4_2_3",
                "decode_0",
                "decode_1",
                "decode_2",
                "decode_3",
                "decode_4",
                "decode_5",
            ]

        self.IN_LAYER_PROXY_SPEC = \
            {
                "stem_2": "layer1_1",
                "stem_5": "layer1_1",
                "stem_8": "layer1_1",

                "layer1_0_1": "layer1_2",
                "layer1_0_2": "layer1_2",
                "layer1_0_3": "layer1_2",

                "layer1_1_1": "layer2_0",
                "layer1_1_2": "layer2_0",
                "layer1_1_3": "layer2_0",

                "layer1_2_1": "layer2_1",
                "layer1_2_2": "layer2_1",
                "layer1_2_3": "layer2_1",

                "layer2_0_1": "layer2_2",
                "layer2_0_2": "layer2_2",
                "layer2_0_3": "layer2_2",

                "layer2_1_1": "layer2_3",
                "layer2_1_2": "layer2_3",
                "layer2_1_3": "layer2_3",

                "layer2_2_1": "layer3_0",
                "layer2_2_2": "layer3_0",
                "layer2_2_3": "layer3_0",

                "layer2_3_1": "layer3_1",
                "layer2_3_2": "layer3_1",
                "layer2_3_3": "layer3_1",

                "layer3_0_1": "layer3_2",
                "layer3_0_2": "layer3_2",
                "layer3_0_3": "layer3_2",

                "layer3_1_1": "layer3_3",
                "layer3_1_2": "layer3_3",
                "layer3_1_3": "layer3_3",

                "layer3_2_1": "layer3_4",
                "layer3_2_2": "layer3_4",
                "layer3_2_3": "layer3_4",

                "layer3_3_1": "layer3_5",
                "layer3_3_2": "layer3_5",
                "layer3_3_3": "layer3_5",

                "layer3_4_1": "layer4_0",
                "layer3_4_2": "layer4_0",
                "layer3_4_3": "layer4_0",

                "layer3_5_1": "layer4_1",
                "layer3_5_2": "layer4_1",
                "layer3_5_3": "layer4_1",

                "layer4_0_1": "layer4_2",
                "layer4_0_2": "layer4_2",
                "layer4_0_3": "layer4_2",

                "layer4_1_1": "decode",
                "layer4_1_2": "decode",
                "layer4_1_3": "decode",

                "layer4_2_1": "decode",
                "layer4_2_2": "decode",
                "layer4_2_3": "decode",

                "decode_0": None,
                "decode_1": None,
                "decode_2": None,
                "decode_3": None,
                "decode_4": None,
                "decode_5": None,
            }

        self.LAYER_NAME_TO_CHANNELS = \
            {'stem_2': 32, 'stem_5': 32, 'stem_8': 64, 'layer1_0_1': 64, 'layer1_0_2': 64,
             'layer1_0_3': 256, 'layer1_1_1': 64, 'layer1_1_2': 64, 'layer1_1_3': 256, 'layer1_2_1': 64,
             'layer1_2_2': 64, 'layer1_2_3': 256, 'layer2_0_1': 128, 'layer2_0_2': 128, 'layer2_0_3': 512,
             'layer2_1_1': 128, 'layer2_1_2': 128, 'layer2_1_3': 512, 'layer2_2_1': 128, 'layer2_2_2': 128,
             'layer2_2_3': 512, 'layer2_3_1': 128, 'layer2_3_2': 128, 'layer2_3_3': 512, 'layer3_0_1': 256,
             'layer3_0_2': 256, 'layer3_0_3': 1024, 'layer3_1_1': 256, 'layer3_1_2': 256,
             'layer3_1_3': 1024, 'layer3_2_1': 256, 'layer3_2_2': 256, 'layer3_2_3': 1024,
             'layer3_3_1': 256, 'layer3_3_2': 256, 'layer3_3_3': 1024, 'layer3_4_1': 256,
             'layer3_4_2': 256, 'layer3_4_3': 1024, 'layer3_5_1': 256, 'layer3_5_2': 256,
             'layer3_5_3': 1024, 'layer4_0_1': 512, 'layer4_0_2': 512, 'layer4_0_3': 2048,
             'layer4_1_1': 512, 'layer4_1_2': 512, 'layer4_1_3': 2048, 'layer4_2_1': 512,
             'layer4_2_2': 512, 'layer4_2_3': 2048, 'decode_0': 512, 'decode_1': 512, 'decode_2': 512,
             'decode_3': 512, 'decode_4': 512, 'decode_5': 512}

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                "stem_2": BLOCK_SIZES_FULL,
                "stem_5": BLOCK_SIZES_FULL,
                "stem_8": BLOCK_SIZES_FULL,
                "layer1_0_1": BLOCK_SIZES_FULL,
                "layer1_0_2": BLOCK_SIZES_FULL,
                "layer1_0_3": BLOCK_SIZES_FULL,
                "layer1_1_1": BLOCK_SIZES_FULL,
                "layer1_1_2": BLOCK_SIZES_FULL,
                "layer1_1_3": BLOCK_SIZES_FULL,
                "layer1_2_1": BLOCK_SIZES_FULL,
                "layer1_2_2": BLOCK_SIZES_FULL,
                "layer1_2_3": BLOCK_SIZES_FULL,
                "layer2_0_1": BLOCK_SIZES_FULL,
                "layer2_0_2": BLOCK_SIZES_FULL,
                "layer2_0_3": BLOCK_SIZES_FULL,
                "layer2_1_1": BLOCK_SIZES_FULL,
                "layer2_1_2": BLOCK_SIZES_FULL,
                "layer2_1_3": BLOCK_SIZES_FULL,
                "layer2_2_1": BLOCK_SIZES_FULL,
                "layer2_2_2": BLOCK_SIZES_FULL,
                "layer2_2_3": BLOCK_SIZES_FULL,
                "layer2_3_1": BLOCK_SIZES_FULL,
                "layer2_3_2": BLOCK_SIZES_FULL,
                "layer2_3_3": BLOCK_SIZES_FULL,
                "layer3_0_1": BLOCK_SIZES_FULL,
                "layer3_0_2": BLOCK_SIZES_FULL,
                "layer3_0_3": BLOCK_SIZES_FULL,
                "layer3_1_1": BLOCK_SIZES_FULL,
                "layer3_1_2": BLOCK_SIZES_FULL,
                "layer3_1_3": BLOCK_SIZES_FULL,
                "layer3_2_1": BLOCK_SIZES_FULL,
                "layer3_2_2": BLOCK_SIZES_FULL,
                "layer3_2_3": BLOCK_SIZES_FULL,
                "layer3_3_1": BLOCK_SIZES_FULL,
                "layer3_3_2": BLOCK_SIZES_FULL,
                "layer3_3_3": BLOCK_SIZES_FULL,
                "layer3_4_1": BLOCK_SIZES_FULL,
                "layer3_4_2": BLOCK_SIZES_FULL,
                "layer3_4_3": BLOCK_SIZES_FULL,
                "layer3_5_1": BLOCK_SIZES_FULL,
                "layer3_5_2": BLOCK_SIZES_FULL,
                "layer3_5_3": BLOCK_SIZES_MINI,
                "layer4_0_1": BLOCK_SIZES_FULL,
                "layer4_0_2": BLOCK_SIZES_FULL,
                "layer4_0_3": BLOCK_SIZES_MINI,
                "layer4_1_1": BLOCK_SIZES_FULL,
                "layer4_1_2": BLOCK_SIZES_FULL,
                "layer4_1_3": BLOCK_SIZES_MINI,
                "layer4_2_1": BLOCK_SIZES_FULL,
                "layer4_2_2": BLOCK_SIZES_FULL,
                "layer4_2_3": BLOCK_SIZES_MINI,
                "decode_0": BLOCK_SIZES_MINI,
                "decode_1": BLOCK_SIZES_MINI,
                "decode_2": BLOCK_SIZES_MINI,
                "decode_3": BLOCK_SIZES_MINI,
                "decode_4": BLOCK_SIZES_MINI,
                "decode_5": BLOCK_SIZES_MINI,
            }

        self.LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS = LAYER_NAME_AND_HIERARCHY_LEVEL_TO_NUM_OF_CHANNEL_GROUPS

        self.LAYER_NAME_TO_LAYER_DIM = {'stem_2': 256, 'stem_5': 256, 'stem_8': 256,
                                        'layer1_0_1': 128, 'layer1_0_2': 128, 'layer1_0_3': 128,
                                        'layer1_1_1': 128, 'layer1_1_2': 128, 'layer1_1_3': 128,
                                        'layer1_2_1': 128, 'layer1_2_2': 128, 'layer1_2_3': 128,
                                        'layer2_0_1': 128, 'layer2_0_2': 64, 'layer2_0_3': 64,
                                        'layer2_1_1': 64, 'layer2_1_2': 64, 'layer2_1_3': 64,
                                        'layer2_2_1': 64, 'layer2_2_2': 64, 'layer2_2_3': 64,
                                        'layer2_3_1': 64, 'layer2_3_2': 64, 'layer2_3_3': 64,
                                        'layer3_0_1': 64, 'layer3_0_2': 64, 'layer3_0_3': 64, 'layer3_1_1': 64,
                                        'layer3_1_2': 64, 'layer3_1_3': 64, 'layer3_2_1': 64, 'layer3_2_2': 64,
                                        'layer3_2_3': 64, 'layer3_3_1': 64, 'layer3_3_2': 64, 'layer3_3_3': 64,
                                        'layer3_4_1': 64, 'layer3_4_2': 64, 'layer3_4_3': 64, 'layer3_5_1': 64,
                                        'layer3_5_2': 64, 'layer3_5_3': 64, 'layer4_0_1': 64, 'layer4_0_2': 64,
                                        'layer4_0_3': 64, 'layer4_1_1': 64, 'layer4_1_2': 64, 'layer4_1_3': 64,
                                        'layer4_2_1': 64, 'layer4_2_2': 64, 'layer4_2_3': 64, 'decode_0': 1,
                                        'decode_1': 64, 'decode_2': 64, 'decode_3': 64, 'decode_4': 64, 'decode_5': 64}

        self.LAYER_NAME_TO_RELU_COUNT = {'stem_2': 2097152, 'stem_5': 2097152, 'stem_8': 4194304, 'layer1_0_1': 1048576,
                                         'layer1_0_2': 1048576, 'layer1_0_3': 4194304, 'layer1_1_1': 1048576,
                                         'layer1_1_2': 1048576,
                                         'layer1_1_3': 4194304, 'layer1_2_1': 1048576, 'layer1_2_2': 1048576,
                                         'layer1_2_3': 4194304,
                                         'layer2_0_1': 2097152, 'layer2_0_2': 524288, 'layer2_0_3': 2097152,
                                         'layer2_1_1': 524288,
                                         'layer2_1_2': 524288, 'layer2_1_3': 2097152, 'layer2_2_1': 524288,
                                         'layer2_2_2': 524288,
                                         'layer2_2_3': 2097152, 'layer2_3_1': 524288, 'layer2_3_2': 524288,
                                         'layer2_3_3': 2097152,
                                         'layer3_0_1': 1048576, 'layer3_0_2': 1048576, 'layer3_0_3': 4194304,
                                         'layer3_1_1': 1048576,
                                         'layer3_1_2': 1048576, 'layer3_1_3': 4194304, 'layer3_2_1': 1048576,
                                         'layer3_2_2': 1048576,
                                         'layer3_2_3': 4194304, 'layer3_3_1': 1048576, 'layer3_3_2': 1048576,
                                         'layer3_3_3': 4194304,
                                         'layer3_4_1': 1048576, 'layer3_4_2': 1048576, 'layer3_4_3': 4194304,
                                         'layer3_5_1': 1048576,
                                         'layer3_5_2': 1048576, 'layer3_5_3': 4194304, 'layer4_0_1': 2097152,
                                         'layer4_0_2': 2097152,
                                         'layer4_0_3': 8388608, 'layer4_1_1': 2097152, 'layer4_1_2': 2097152,
                                         'layer4_1_3': 8388608,
                                         'layer4_2_1': 2097152, 'layer4_2_2': 2097152, 'layer4_2_3': 8388608,
                                         'decode_0': 512,
                                         'decode_1': 2097152, 'decode_2': 2097152, 'decode_3': 2097152,
                                         'decode_4': 2097152,
                                         'decode_5': 2097152}

        self.LAYER_HIERARCHY_SPEC = LAYER_HIERARCHY_SPEC

        self.HIERARCHY_NAME = HIERARCHY_NAME


class ParamsFactory:
    def __init__(self):
        self.classes = {
            "ResNetParams": ResNetParams,
            "MobileNetV2Params": MobileNetV2Params,
        }

    def __call__(self, json_file_name):
        with open(json_file_name, 'rb') as f:
            content = json.load(f)
        type = content['type']
        del content['type']
        return self.classes[type](**content)

# params = ParamsFactory()("/home/yakir/PycharmProjects/secure_inference/research/block_relu/distortion_handler_configs/resnet_COCO_164K_2_hierarchies.json")
#
# params.HIERARCHY_NAME
