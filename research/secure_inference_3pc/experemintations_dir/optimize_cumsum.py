import torch
import numpy as np

from research.communication.utils import Sender, Receiver
import time
from numba import njit, prange

# @njit(parallel=True)
def cumsum2(w):
    out = np.zeros_like(w)
    for i in range(1, 64):
        out[:,i] = out[:,i -1] + w[:, i]
    return out

def cumsum3(w):
    w_tmp = w.copy()
    view = np.flip(w, 1)
    np.cumsum(view, 1, out=view)
    view = np.flip(w, 1)
    return np.subtract(view, w_tmp, out=view)
    # out = np.zeros_like(w)
    # for i in range(1, 64):
    #     out[:,i] = out[:,i -1] + w[:, i]
    # return out

def cumsum(w):
    return w[..., ::-1].cumsum(axis=-1)[..., ::-1] - w

private_prf_numpy = np.random.default_rng(seed=31243)
dtype = np.int32
value_0 = private_prf_numpy.integers(low=-65, high=67, dtype=dtype, size=(5000000, 64))

t0 = time.time()
r = cumsum3(value_0)
print(time.time() - t0)

