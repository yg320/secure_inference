from research.distortion.parameters.block_sizes import BLOCK_SIZES_32x32, BLOCK_SIZES_16x16, BLOCK_SIZES_8x8, BLOCK_SIZES_4x4

class Params:
    def __init__(self):

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
        ]

        self.LAYER_NAMES = \
            [
                "stem",
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
            ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                "stem": "stem",
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
            }

        self.BLOCK_INPUT_DICT = \
            {
                'stem': "input_images",
                "layer1_0": "stem",
                "layer1_1": "layer1_0",
                "layer2_0": "layer1_1",
                "layer2_1": "layer2_0",
                "layer3_0": "layer2_1",
                "layer3_1": "layer3_0",
                "layer4_0": "layer3_1",
                "layer4_1": "layer4_0",
            }

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'stem': BLOCK_SIZES_32x32,
                'layer1_0_1': BLOCK_SIZES_32x32,
                'layer1_0_2': BLOCK_SIZES_32x32,
                'layer1_1_1': BLOCK_SIZES_32x32,
                'layer1_1_2': BLOCK_SIZES_32x32,
                'layer2_0_1': BLOCK_SIZES_16x16,
                'layer2_0_2': BLOCK_SIZES_16x16,
                'layer2_1_1': BLOCK_SIZES_16x16,
                'layer2_1_2': BLOCK_SIZES_16x16,
                'layer3_0_1': BLOCK_SIZES_8x8,
                'layer3_0_2': BLOCK_SIZES_8x8,
                'layer3_1_1': BLOCK_SIZES_8x8,
                'layer3_1_2': BLOCK_SIZES_8x8,
                'layer4_0_1': BLOCK_SIZES_4x4,
                'layer4_0_2': BLOCK_SIZES_4x4,
                'layer4_1_1': BLOCK_SIZES_4x4,
                'layer4_1_2': BLOCK_SIZES_4x4
            }

        self.LAYER_NAME_TO_DIMS = \
            {
                'stem':       [8, 32, 32],
                'layer1_0_1': [32, 16, 16],
                'layer1_0_2': [32, 16, 16],
                'layer1_1_1': [32, 16, 16],
                'layer1_1_2': [32, 16, 16],
                'layer2_0_1': [64, 8, 8],
                'layer2_0_2': [64, 8, 8],
                'layer2_1_1': [64, 8, 8],
                'layer2_1_2': [64, 8, 8],
                'layer3_0_1': [128, 4, 4],
                'layer3_0_2': [128, 4, 4],
                'layer3_1_1': [128, 4, 4],
                'layer3_1_2': [128, 4, 4],
                'layer4_0_1': [256, 2, 2],
                'layer4_0_2': [256, 2, 2],
                'layer4_1_1': [256, 2, 2],
                'layer4_1_2': [256, 2, 2],
            }  # 69632
