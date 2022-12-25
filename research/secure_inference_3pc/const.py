import numpy as np
import torch


CLIENT = 0
SERVER = 1
CRYPTO_PROVIDER = 2
P = 67
NUM_BITS = 64
TRUNC = 500000

UNSIGNED_DTYPE = np.ulonglong
SIGNED_DTYPE = np.int64
TORCH_DTYPE = torch.int64

MIN_VAL = np.iinfo(np.int64).min
MAX_VAL = np.iinfo(np.int64).max

