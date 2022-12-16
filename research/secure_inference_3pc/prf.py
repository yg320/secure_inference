import numpy as np
from typing import Tuple
import torch
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER
from threading import Thread
import queue
import time
import pickle

# TODO: https://towardsdatascience.com/six-levels-of-python-decorators-1f12c9067b23


class ThreadedPRFWrapper(Thread):
    def __init__(self, random_spec, seed):
        super(ThreadedPRFWrapper, self).__init__()

        self.random_spec = random_spec
        self.seed = seed
        self.prf = np.random.default_rng(seed=seed)
        self.non_threaded_prf = np.random.default_rng(seed=seed)  # TODO: change seed to random.number something
        self.random_tensor_queue = queue.Queue(maxsize=10)

    def run(self):

        for spec in self.random_spec:
            low, high, size, dtype = spec
            out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
            self.random_tensor_queue.put(out)

    def integers(self, *args, **kwargs):
        with Timer("PRF - integers"):
            out = self.random_tensor_queue.get()
        return out

    def permutation(self, data, axis):
        out = self.non_threaded_prf.permutation(x=data, axis=axis)
        return out

    def get_random_tensor_over_L(self, shape):
        dtype = np.int64
        rand = self.integers(low=np.iinfo(dtype).min // 2,
                             high=np.iinfo(dtype).max // 2,
                             size=shape,
                             dtype=dtype)
        return torch.from_numpy(rand)


class PRFWrapper:
    def __init__(self, seed, random_spec=None):
        self.prf = np.random.default_rng(seed=seed)

        self.calls_monitor = []

    def integers(self, low, high, size, dtype):
        self.calls_monitor.append((low, high, size, dtype))
        out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
        return out

    def permutation(self, data, axis):
        out = self.prf.permutation(x=data, axis=axis)
        return out

    def get_random_tensor_over_L(self, shape):
        dtype = np.int64
        rand = self.integers(low=np.iinfo(dtype).min // 2,
                             high=np.iinfo(dtype).max // 2,
                             size=tuple(shape),
                             dtype=dtype)
        return torch.from_numpy(rand)


class MultiPartyPRFHandler:
    def __init__(self, seeds):
        self.random_specs = {
            (CLIENT, SERVER): pickle.load(file=open("/home/yakir/01.pickle", 'rb')),
            (CLIENT, CRYPTO_PROVIDER): pickle.load(file=open("/home/yakir/02.pickle", 'rb')),
            (SERVER, CRYPTO_PROVIDER): pickle.load(file=open("/home/yakir/12.pickle", 'rb')),
            CLIENT: pickle.load(file=open("/home/yakir/0.pickle", 'rb')),
            SERVER: pickle.load(file=open("/home/yakir/1.pickle", 'rb')),
            CRYPTO_PROVIDER: pickle.load(file=open("/home/yakir/2.pickle", 'rb')),

        }
        self.prfs = {parties: ThreadedPRFWrapper(random_spec=self.random_specs[parties], seed=seed) for parties, seed in seeds.items()}
        for val in self.prfs.values():
            val.start()

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

    with Timer("PRF"):
        out = prf_handler[(CLIENT, SERVER)].integers(1, high=67, size=(884736, 64), dtype=np.int64)

    with Timer("NUMPY"):
        x = np.arange(884736*64)
        # out = prf_handler[(CLIENT, SERVER)].integers(1, high=67, size=(884736, 64), dtype=np.int64)
    print(out)

