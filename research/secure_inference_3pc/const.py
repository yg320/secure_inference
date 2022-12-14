import numpy as np
import torch


CLIENT = 0
SERVER = 1
CRYPTO_PROVIDER = 2
P = 67
NUM_BITS = 64
NUM_OF_COMPARE_BITS = 28
IGNORE_MSB_BITS = 8

TRUNC_BITS = 16
TRUNC = 2 ** TRUNC_BITS
IS_TORCH_BACKEND = True

UNSIGNED_DTYPE = {32: np.uint32, 64: np.uint64}[NUM_BITS]
SIGNED_DTYPE = {32: np.int32, 64: np.int64}[NUM_BITS]
TORCH_DTYPE = {32: torch.int32, 64: torch.int64}[NUM_BITS]

MIN_VAL = np.iinfo(SIGNED_DTYPE).min
MAX_VAL = np.iinfo(SIGNED_DTYPE).max


