import numpy as np
import time
import torch
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.base import decompose_torch, get_c_party_0_torch, get_c_party_0, decompose

IGNORE_MSB_BITS = 0
NUM_BITS = 64
NUM_OF_COMPARE_BITS = 64
P = 67
UNSIGNED_DTYPE = np.uint64
min_org_shit = -283206
max_org_shit = 287469
org_shit = (np.arange(min_org_shit, max_org_shit + 1) % P).astype(np.uint8)


def module_67(xxx):
    orig_shape = xxx.shape
    xxx = xxx.reshape(-1)
    np.subtract(xxx, min_org_shit, out=xxx)
    return org_shit[xxx].reshape(orig_shape)

powers = np.arange(NUM_BITS, dtype=UNSIGNED_DTYPE)[np.newaxis][:,::-1]
powers_torch = torch.from_numpy(powers.astype(np.int64)).to("cuda:0")

# def decompose_torch(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#
#     r_shift = value >> powers_torch[:, NUM_BITS - NUM_OF_COMPARE_BITS:]
#     value_bits = r_shift & 1
#
#     ret = value_bits.to(torch.int8).reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret

# def decompose(value, out=None, out_mask=None):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#     end = None if IGNORE_MSB_BITS == 0 else -IGNORE_MSB_BITS
#     r_shift = value >> powers[:,NUM_BITS - NUM_OF_COMPARE_BITS-IGNORE_MSB_BITS:end]
#     value_bits = np.zeros(shape=(value.shape[0], NUM_OF_COMPARE_BITS), dtype=np.int8)
#     np.bitwise_and(r_shift, np.int8(1), out=value_bits)
#     ret = value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret
#
# def get_c_party_0(x_bits, multiplexer_bits, beta, j):
#     beta = beta[..., np.newaxis]
#     beta = 2 * beta  # Not allowed to change beta inplace
#     np.subtract(beta, 1, out=beta)
#     np.multiply(multiplexer_bits, x_bits, out=multiplexer_bits)
#     np.multiply(multiplexer_bits, -2, out=multiplexer_bits)
#     np.add(multiplexer_bits, x_bits, out=multiplexer_bits)
#
#     w_cumsum = multiplexer_bits.astype(np.int32)
#     np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
#     np.subtract(w_cumsum, multiplexer_bits, out=w_cumsum)
#     np.multiply(x_bits, beta, out=x_bits)
#     np.add(w_cumsum, x_bits, out=w_cumsum)
#
#     return w_cumsum

# def get_c_party_0_torch(x_bits, multiplexer_bits, beta, j):
#
#     beta = beta[..., np.newaxis]
#     beta = 2 * beta  # Not allowed to change beta inplace
#     torch.sub(beta, 1, out=beta)
#     torch.mul(multiplexer_bits, x_bits, out=multiplexer_bits)
#     torch.mul(multiplexer_bits, -2, out=multiplexer_bits)
#     torch.add(multiplexer_bits, x_bits, out=multiplexer_bits)
#
#     w_cumsum = multiplexer_bits.to(torch.int32)
#     torch.cumsum(w_cumsum, dim=-1, out=w_cumsum)
#     # np.cumsum(w_cumsum, axis=-1, out=w_cumsum)
#     torch.sub(w_cumsum, multiplexer_bits, out=w_cumsum)
#     torch.mul(x_bits, beta, out=x_bits)
#     torch.add(w_cumsum, x_bits, out=w_cumsum)
#
#     return w_cumsum

N = 10000
x_bits_0 = np.random.randint(low=0, high=P, size=(N, NUM_BITS), dtype=np.int8)
r = np.random.randint(low=np.iinfo(np.uint64).min, high=np.iinfo(np.uint64).max, size=(N,), dtype=np.uint64)
beta = np.random.randint(low=0, high=2, size=(N,), dtype=np.int8)
s = np.random.randint(low=1, high=P, size=x_bits_0.shape, dtype=np.int32)

r = r.astype(np.int64)
with Timer("Torch"):
    r_torch = torch.from_numpy(r).to("cuda:0")
    x_bits_0_torch = torch.from_numpy(x_bits_0).to("cuda:0")
    beta_torch = torch.from_numpy(beta).to("cuda:0")
    s_torch = torch.from_numpy(s).to("cuda:0")

    r_torch[beta_torch.to(torch.int64)] += 1
    bits_torch = decompose_torch(r_torch)
    c_bits_0_torch = get_c_party_0_torch(x_bits_0_torch, bits_torch, beta_torch)
    torch.mul(s_torch, c_bits_0_torch, out=s_torch)
    d_bits_0_torch = s_torch % 67
    d_bits_0_torch = d_bits_0_torch.cpu().numpy().astype(np.uint8)
    # d_bits_0_torch_shuffled = d_bits_0_torch[:, torch.randperm(d_bits_0_torch.size()[1]) ]
    # d_bits_0_torch_shuffled.to("cpu").numpy()

# prf = np.random.default_rng(seed=0)
with Timer("Numpy"):
    r[beta] += 1
    bits = decompose(r)
    c_bits_0 = get_c_party_0(x_bits_0, bits, beta, np.int8(0))
    np.multiply(s, c_bits_0, out=s)
    d_bits_0 = module_67(s)
    # d_bits_0 = prf.permutation(d_bits_0, axis=-1)
