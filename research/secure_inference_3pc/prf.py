import numpy as np
import torch
from research.secure_inference_3pc.timer import Timer
from threading import Thread
import queue
from research.secure_inference_3pc.const import IS_TORCH_BACKEND
from research.secure_inference_3pc.timer import Timer
# TODO: https://towardsdatascience.com/six-levels-of-python-decorators-1f12c9067b23
# dtype_converted = {
#     np.int8: torch.int8,
#     torch.int8: torch.int8,
#
#
#     np.int32: torch.int32,
#     torch.int32: torch.int32,
#
#     np.int64: torch.int64,
#     torch.int64: torch.int64
# }
dtype_converted = {
    np.int8: np.int8,
    torch.int8: np.int8,


    np.int32: np.int32,
    torch.int32: np.int32,

    np.int64: np.int64,
    torch.int64: np.int64,
    np.dtype("int64"): np.int64,
    np.dtype("int8"): np.int8,
    np.dtype("int32"): np.int32
}
class PRFWrapper:
    def __init__(self, seed, queue, device):
        self.device = device
        self.seed = seed
        self.prf = np.random.default_rng(seed=self.seed)
        self.non_threaded_prf = np.random.default_rng(seed=self.seed)  # TODO: change seed to random.number something
        self.torch_prf = torch.Generator().manual_seed(self.seed)
        self.queue = queue
        self.fetch = False

    def integers_fetch(self, low, high, size, dtype):
        out = self.prf.integers(low=low, high=high, size=size, dtype=dtype_converted[dtype])
        # print("fetch:", low, high, size, dtype)
        self.queue.put(out)

    def integers(self, low, high, size, dtype):
        if self.fetch:
            # with Timer("Integers - fetch"):
            ret = self.queue.get()
            assert ret.shape == tuple(size), f"{ret.shape} , {tuple(size)}"
            assert dtype_converted[ret.dtype] == dtype_converted[dtype], f"{ret.dtype} , {dtype}"
            if IS_TORCH_BACKEND:
                ret = torch.from_numpy(ret).to(self.device)
            return ret
        else:
            out = self.prf.integers(low=low, high=high, size=size, dtype=dtype_converted[dtype])

            if IS_TORCH_BACKEND:
                out = torch.from_numpy(out).to(self.device)
            return out

    def permutation(self, data, axis):
        if IS_TORCH_BACKEND:
            return data[:, torch.randperm(data.size()[axis], generator=self.torch_prf)]
        else:
            out = self.non_threaded_prf.permutation(x=data, axis=axis)
            return out



class MultiPartyPRFHandler(Thread):
    def __init__(self, party, seeds, device):
        super(MultiPartyPRFHandler, self).__init__()
        self.party = party
        self.random_tensor_queue = queue.Queue(maxsize=20)
        self.device = device
        self.prfs = {parties: PRFWrapper(seed=seed, queue=self.random_tensor_queue, device=self.device) for parties, seed in seeds.items()}

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

