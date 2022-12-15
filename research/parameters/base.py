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
     [0, 1],
     [1, 0]
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
    [64, 64],
    [0, 1],
    [1, 0]
]

BLOCK_SIZES_96x96 = [
    [1, 1],
    [1, 2],
    [2, 1],
    [1, 3],
    [3, 1],
    [1, 4],
    [2, 2],
    [4, 1],
    [1, 6],
    [2, 3],
    [3, 2],
    [6, 1],
    [2, 4],
    [4, 2],
    [3, 3],
    [1, 12],
    [2, 6],
    [3, 4],
    [4, 3],
    [6, 2],
    [12, 1],
    [4, 4],
    [3, 6],
    [6, 3],
    [2, 12],
    [4, 6],
    [6, 4],
    [12, 2],
    [3, 12],
    [6, 6],
    [12, 3],
    [4, 12],
    [12, 4],
    [6, 12],
    [12, 6],
    [12, 12],
    [0, 1],
    [1, 0]
]

BLOCK_SIZES_192x192 = [
    [1, 1], [1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [2, 2], [4, 1], [1, 6], [2, 3], [3, 2], [6, 1],
    [1, 8], [2, 4], [4, 2], [8, 1], [3, 3], [1, 12], [2, 6], [3, 4], [4, 3], [6, 2], [12, 1], [2, 8],
    [4, 4], [8, 2], [3, 6], [6, 3], [1, 24], [2, 12], [3, 8], [4, 6], [6, 4], [8, 3], [12, 2],
    [24, 1], [4, 8], [8, 4], [3, 12], [6, 6], [12, 3], [2, 24], [4, 12], [6, 8], [8, 6], [12, 4],
    [24, 2], [8, 8], [3, 24], [6, 12], [12, 6], [24, 3], [4, 24], [8, 12], [12, 8], [24, 4], [6, 24],
    [12, 12], [24, 6], [8, 24], [24, 8], [12, 24], [24, 12], [24, 24], [0, 1], [1, 0]
]

BLOCK_SIZES_256x256 = [[1, 1], [1, 2], [2, 1], [1, 4], [2, 2], [4, 1], [1, 8], [2, 4], [4, 2], [8, 1], [1, 16], [2, 8],
                       [4, 4], [8, 2], [16, 1], [1, 32], [2, 16], [4, 8], [8, 4], [16, 2], [32, 1], [2, 32], [4, 16],
                       [8, 8], [16, 4], [32, 2], [4, 32], [8, 16], [16, 8], [32, 4], [8, 32], [16, 16], [32, 8],
                       [16, 32], [32, 16], [32, 32], [0, 1], [1, 0]]
