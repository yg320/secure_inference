import numpy as np
from typing import Tuple
import torch

class ThreeMPCPRR:
    def __init__(self, seed):
        self.prf = np.random.default_rng(seed=seed)

    def integers(self, low, high, size, dtype):
        return self.prf.integers(low=low, high=high, size=size, dtype=dtype)

    def permutation(self, data, axis):
        return self.prf.permutation(x=data, axis=axis)

    def get_random_tensor_over_L(self, shape):
        dtype = np.int64
        rand = self.integers(low=np.iinfo(dtype).min // 2,
                             high=np.iinfo(dtype).max // 2,
                             size=shape,
                             dtype=dtype)
        return torch.from_numpy(rand)

class MultiPartyPRFHandler:
    def __init__(self, seeds):
        self.prfs = {parties: ThreeMPCPRR(seed=seed) for parties, seed in seeds.items()}

    def __getitem__(self, parties: Tuple):
        if parties in self.prfs.keys():
            return self.prfs[parties]
        else:
            return self.prfs[parties[::-1]]



if __name__ == "__main__":
    from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER

    prf_handler = MultiPartyPRFHandler({
        (CLIENT, SERVER): 0,
        (CLIENT, CRYPTO_PROVIDER): 1,
        (SERVER, CRYPTO_PROVIDER): None
    })

    print(prf_handler[CLIENT, SERVER].integers(0, 2, (100, ), np.uint8))