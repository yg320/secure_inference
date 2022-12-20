import numpy as np
from typing import Tuple
import torch
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from threading import Thread
import queue
import time
import pickle
import os

# TODO: https://towardsdatascience.com/six-levels-of-python-decorators-1f12c9067b23

class PRFWrapper:
    def __init__(self, seed, queue):
        self.prf = np.random.default_rng(seed=seed)

    def integers(self, low, high, size, dtype):
        out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
        return out

    def permutation(self, data, axis):
        out = self.prf.permutation(x=data, axis=axis)
        return out

    # def get_random_tensor_over_L(self, shape):
    #     dtype = np.int64
    #     rand = self.integers(low=np.iinfo(dtype).min // 2,
    #                          high=np.iinfo(dtype).max // 2,
    #                          size=tuple(shape),
    #                          dtype=dtype)
    #     return torch.from_numpy(rand)


class MultiPartyPRFHandler(Thread):
    def __init__(self, party, seeds):
        super(MultiPartyPRFHandler, self).__init__()
        self.party = party
        self.random_tensor_queue = queue.Queue(maxsize=10)

        self.prfs = {parties: PRFWrapper(seed=seed, queue=self.random_tensor_queue) for parties, seed in seeds.items()}

    def fetch(self, repeat, model, image):
        self.repeat = repeat
        self.model = model
        self.image = image

        for v in self.prfs.values():
            v.fetch = True

        self.start()

    def __getitem__(self, parties):
        if parties in self.prfs.keys():
            return self.prfs[parties]
        else:
            return self.prfs[parties[::-1]]

    def run(self):
        for _ in range(self.repeat):
            self.model(self.image)

