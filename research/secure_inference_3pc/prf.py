import numpy as np
import torch
from threading import Thread
import threading
import queue
import time

class PRFWrapper:
    def __init__(self, seed, queue, device):
        self.device = device
        self.seed = seed
        self.prf = np.random.default_rng(seed=self.seed)
        self.non_threaded_prf = np.random.default_rng(seed=self.seed)  # TODO: change seed to random.number something
        self.torch_prf = torch.Generator().manual_seed(self.seed)
        self.queue = queue
        self.fetch = False
        self.lock = threading.Lock()

    def integers_fetch(self, low, high, size, dtype):
        out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
        self.queue.put(out)

    def integers(self, low, high, size, dtype):
        if self.fetch:

            ret = self.queue.get()
            assert ret.shape == tuple(size), f"{ret.shape} , {tuple(size)}"
            assert ret.dtype == dtype, f"{ret.dtype} , {dtype}"

            return ret
        else:
            out = self.prf.integers(low=low, high=high, size=size, dtype=dtype)
            return out

    def permutation(self, data, axis):
        out = self.non_threaded_prf.permutation(x=data, axis=axis)
        return out


class MultiPartyPRFHandler(Thread):
    def __init__(self, party, seeds, device):
        super(MultiPartyPRFHandler, self).__init__()
        self.party = party
        self.random_tensor_queue = queue.Queue(maxsize=20)
        self.device = device
        self.prfs = {parties: PRFWrapper(seed=seed, queue=self.random_tensor_queue, device=self.device) for parties, seed in seeds.items()}
        self.lock = threading.Lock()

        self.image_shape = None
        self.stop = False

    def fetch(self, model):
        self.model = model

        for v in self.prfs.values():
            v.fetch = True

        self.start()

    def fetch_image(self, image_shape):
        # TODO: use with
        self.lock.acquire()
        self.image_shape = image_shape
        self.lock.release()

    def __getitem__(self, parties):
        if parties in self.prfs.keys():
            return self.prfs[parties]
        else:
            return self.prfs[parties[::-1]]

    def done(self):
        self.lock.acquire()
        self.stop = True
        self.lock.release()

    def run(self):

        while True:
            self.lock.acquire()
            stop = self.stop
            self.lock.release()

            if stop:
                break

            self.lock.acquire()
            image_shape = self.image_shape
            self.image_shape = None
            self.lock.release()
            if image_shape is None:
                time.sleep(0.01)
            else:

                self.model(image_shape)


