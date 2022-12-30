from research.distortion.parameters.block_sizes import BLOCK_SIZES_112x112, BLOCK_SIZES_56x56, BLOCK_SIZES_28x28, BLOCK_SIZES_14x14, BLOCK_SIZES_7x7

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
            None
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
                None: "layer4_1"
            }

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'stem': BLOCK_SIZES_112x112,
                'layer1_0_1': BLOCK_SIZES_56x56,
                'layer1_0_2': BLOCK_SIZES_56x56,
                'layer1_1_1': BLOCK_SIZES_56x56,
                'layer1_1_2': BLOCK_SIZES_56x56,
                'layer2_0_1': BLOCK_SIZES_28x28,
                'layer2_0_2': BLOCK_SIZES_28x28,
                'layer2_1_1': BLOCK_SIZES_28x28,
                'layer2_1_2': BLOCK_SIZES_28x28,
                'layer3_0_1': BLOCK_SIZES_14x14,
                'layer3_0_2': BLOCK_SIZES_14x14,
                'layer3_1_1': BLOCK_SIZES_14x14,
                'layer3_1_2': BLOCK_SIZES_14x14,
                'layer4_0_1': BLOCK_SIZES_7x7,
                'layer4_0_2': BLOCK_SIZES_7x7,
                'layer4_1_1': BLOCK_SIZES_7x7,
                'layer4_1_2': BLOCK_SIZES_7x7
            }

        self.LAYER_NAME_TO_DIMS = \
            {
                'stem': [64, 112, 112],
                'layer1_0_1': [64, 56, 56],
                'layer1_0_2': [64, 56, 56],
                'layer1_1_1': [64, 56, 56],
                'layer1_1_2': [64, 56, 56],
                'layer2_0_1': [128, 28, 28],
                'layer2_0_2': [128, 28, 28],
                'layer2_1_1': [128, 28, 28],
                'layer2_1_2': [128, 28, 28],
                'layer3_0_1': [256, 14, 14],
                'layer3_0_2': [256, 14, 14],
                'layer3_1_1': [256, 14, 14],
                'layer3_1_2': [256, 14, 14],
                'layer4_0_1': [512, 7, 7],
                'layer4_0_2': [512, 7, 7],
                'layer4_1_1': [512, 7, 7],
                'layer4_1_2': [512, 7, 7]
            }
