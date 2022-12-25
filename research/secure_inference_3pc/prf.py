import numpy as np
import torch
from research.secure_inference_3pc.timer import Timer
from threading import Thread
import queue


# TODO: https://towardsdatascience.com/six-levels-of-python-decorators-1f12c9067b23

class PRFWrapper:
    def __init__(self, seed, queue):
        self.prf = np.random.default_rng(seed=seed)
        self.non_threaded_prf = np.random.default_rng(seed=seed)  # TODO: change seed to random.number something
        self.queue = queue
        self.fetch = False

    def integers_fetch(self, low, high, size, dtype):
        out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
        # print("fetch:", low, high, size, dtype)
        self.queue.put(out)

    def integers(self, low, high, size, dtype):
        if self.fetch:
            # with Timer("Integers - fetch"):
            ret = self.queue.get()
            assert ret.shape == tuple(size), f"{ret.shape} , {tuple(size)}"
            assert ret.dtype == dtype, f"{ret.dtype} , {dtype}"
            return ret
        else:
            return self.prf.integers(low=low, high=high, size=size, dtype=dtype)

    def permutation(self, data, axis):
        out = self.non_threaded_prf.permutation(x=data, axis=axis)
        return out



class MultiPartyPRFHandler(Thread):
    def __init__(self, party, seeds):
        super(MultiPartyPRFHandler, self).__init__()
        self.party = party
        self.random_tensor_queue = queue.Queue(maxsize=20)

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

