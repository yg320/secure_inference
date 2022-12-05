# import torch
import numpy as np
import random
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

num_bit_to_dtype = {
    8: np.ubyte,
    16: np.ushort,
    32: np.uintc,
    64: np.ulonglong
}
num_bits = 32
dtype = num_bit_to_dtype[num_bits]
min_val = np.iinfo(dtype).min
max_val = np.iinfo(dtype).max

L_minus_1 = dtype(2 ** num_bits - 1)
L = 2 ** num_bits


def add_mode_L_minus_one(a, b):
    ret = a + b
    ret[ret < a] += dtype(1)
    ret[ret == L_minus_1] = dtype(0)
    return ret


def sub_mode_L_minus_one(a, b):
    ret = a - b
    ret[b > a] -= dtype(1)
    return ret

for seed in tqdm(range(10000)):

    rng = np.random.default_rng(seed=0)
    a_0 = rng.integers(min_val, max_val, size=(seed,), dtype=dtype)  # np.random.randint(min_val, max_val + 1, dtype=dtype)
    a_1 = rng.integers(min_val, max_val, size=(seed,), dtype=dtype)
    a = add_mode_L_minus_one(a_0, a_1)
    # int(bin((int(a_0) + int(a_1)) % (2**32 - 1))[2:].zfill(32)[0])
    # (((int(y_0) + int(y_1)) % (2**32 - 1)) % 2)
    # (r%2) ^ (x%2) ^ (x > r)
    beta = rng.integers(0, 2, size=(a_0.size,), dtype=dtype)

    # a_0 = np.array([14], dtype=dtype)
    # a_1 = np.array([15], dtype=dtype)
    # beta = np.array([0], dtype=dtype)
    x = rng.integers(min_val, max_val, size=(a_0.size,), dtype=dtype)
    x_0 = rng.integers(min_val, max_val, size=(a_0.size,), dtype=dtype)  # P2 -> P0
    x_1 = sub_mode_L_minus_one(x, x_0)

    x_bit0 = x % 2
    x_bit_0_0 = rng.integers(min_val, max_val + 1, size=(a_0.size,), dtype=dtype)
    x_bit_0_1 = x_bit0 - x_bit_0_0

    y_0 = add_mode_L_minus_one(a_0, a_0)
    y_1 = add_mode_L_minus_one(a_1, a_1)
    r_0 = add_mode_L_minus_one(x_0, y_0)
    r_1 = add_mode_L_minus_one(x_1, y_1)
    r = add_mode_L_minus_one(r_0, r_1)

    beta_p = beta ^ (x > r)
    beta_p_0 = rng.integers(min_val, max_val + 1, size=(a_0.size,), dtype=dtype)
    beta_p_1 = beta_p - beta_p_0

    gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
    gamma_1 = beta_p_1 + (1 * beta) - (2 * beta * beta_p_1)

    # 8)
    delta_0 = x_bit_0_0 + (0 * (r % 2)) - (2 * (r % 2) * x_bit_0_0)
    delta_1 = x_bit_0_1 + (1 * (r % 2)) - (2 * (r % 2) * x_bit_0_1)

    theta = (gamma_0 + gamma_1) * (delta_0 + delta_1)

    theta_0 = rng.integers(min_val, max_val + 1, size=(a_0.size,), dtype=dtype)
    theta_1 = theta - theta_0


    alpha_0 = gamma_0 + delta_0 - 2 * theta_0
    alpha_1 = gamma_1 + delta_1 - 2 * theta_1

    alpha = (alpha_0 + alpha_1)
    # print('fds')

    assert np.all(alpha == (add_mode_L_minus_one(a, a) % 2))
    # x = 3
# bin(a[x])[2:].zfill(32)[0] == str(alpha[x])