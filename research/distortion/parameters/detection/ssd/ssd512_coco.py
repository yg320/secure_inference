from research.distortion.parameters.block_sizes import BLOCK_SIZES_512x512, BLOCK_SIZES_256x256, BLOCK_SIZES_128x128, BLOCK_SIZES_64x64, BLOCK_SIZES_32x32

class Params:
    def __init__(self):
        self.BLOCK_NAMES = \
            [
                'block_0',
                'block_1',
                'block_2',
                'block_3',
                'block_4',
                'block_5',
                'block_6',
                'block_7',
                'block_8',
                'block_9',
                'block_10',
                'block_11',
                'block_12',
                'block_13',
                'block_14',
                None
            ]

        self.LAYER_NAME_TO_BLOCK_NAME = \
            {
                'relu_0': "block_0",
                'relu_1': "block_1",
                'relu_2': "block_2",
                'relu_3': "block_3",
                'relu_4': "block_4",
                'relu_5': "block_5",
                'relu_6': "block_6",
                'relu_7': "block_7",
                'relu_8': "block_8",
                'relu_9': "block_9",
                'relu_10': "block_10",
                'relu_11': "block_11",
                'relu_12': "block_12",
                'relu_13': "block_13",
                'relu_14': "block_14",
            }

        self.LAYER_NAMES = \
            [
                'relu_0',
                'relu_1',
                'relu_2',
                'relu_3',
                'relu_4',
                'relu_5',
                'relu_6',
                'relu_7',
                'relu_8',
                'relu_9',
                'relu_10',
                'relu_11',
                'relu_12',
                'relu_13',
                'relu_14',
            ]

        self.BLOCK_INPUT_DICT = \
            {
                'block_0': "input_images",
                'block_1': "block_0",
                'block_2': "block_1",
                'block_3': "block_2",
                'block_4': "block_3",
                'block_5': "block_4",
                'block_6': "block_5",
                'block_7': "block_6",
                'block_8': "block_7",
                'block_9': "block_8",
                'block_10': "block_9",
                'block_11': "block_10",
                'block_12': "block_11",
                'block_13': "block_12",
                'block_14': "block_13",
                None: 'block_14'
            }

        self.LAYER_NAME_TO_BLOCK_SIZES = \
            {
                'relu_0': BLOCK_SIZES_512x512,
                'relu_1': BLOCK_SIZES_512x512,
                'relu_2': BLOCK_SIZES_256x256,
                'relu_3': BLOCK_SIZES_256x256,
                'relu_4': BLOCK_SIZES_128x128,
                'relu_5': BLOCK_SIZES_128x128,
                'relu_6': BLOCK_SIZES_128x128,
                'relu_7': BLOCK_SIZES_64x64,
                'relu_8': BLOCK_SIZES_64x64,
                'relu_9': BLOCK_SIZES_64x64,
                'relu_10': BLOCK_SIZES_32x32,
                'relu_11': BLOCK_SIZES_32x32,
                'relu_12': BLOCK_SIZES_32x32,
                'relu_13': BLOCK_SIZES_32x32,
                'relu_14': BLOCK_SIZES_32x32,
            }

        self.LAYER_NAME_TO_DIMS = \
            {
                'relu_0': [64, 512, 512],
                'relu_1': [64, 512, 512],
                'relu_2': [128, 256, 256],
                'relu_3': [128, 256, 256],
                'relu_4': [256, 128, 128],
                'relu_5': [256, 128, 128],
                'relu_6': [256, 128, 128],
                'relu_7': [512, 64, 64],
                'relu_8': [512, 64, 64],
                'relu_9': [512, 64, 64],
                'relu_10': [512, 32, 32],
                'relu_11': [512, 32, 32],
                'relu_12': [512, 32, 32],
                'relu_13': [1024, 32, 32],
                'relu_14': [1024, 32, 32]
            }