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



for i in tqdm(range(1000000)):
    # i = 17087
    # i = 286350
    # i = 123
    rng = np.random.default_rng(seed=0)
    # np.random.seed(i)
    a_0 = rng.integers(min_val, max_val + 1, size=(1000,), dtype=dtype)  # np.random.randint(min_val, max_val + 1, dtype=dtype)
    a_1 = rng.integers(min_val, max_val + 1, size=(1000,), dtype=dtype)
    a = a_0 + a_1
    if (a == L_minus_1).any():
        continue
    rng = np.random.default_rng(seed=123)
    eta_pp = rng.integers(0, 2, size=(a_0.size,), dtype=dtype)

    r = rng.integers(min_val, max_val + 1, size=(a_0.size,), dtype=dtype)
    r_0 = rng.integers(min_val, max_val + 1, size=(a_0.size,), dtype=dtype)
    r_1 = r - r_0
    alpha = (r < r_0).astype(dtype)

    a_tild_0 = a_0 + r_0  # P0 -> P2
    beta_0 = (a_tild_0 < a_0).astype(dtype)

    a_tild_1 = a_1 + r_1  # P1 -> P2
    beta_1 = (a_tild_1 < a_1).astype(dtype)

    x = (a_tild_0 + a_tild_1)
    delta = (x < a_tild_0).astype(dtype)

    delta_0 = rng.integers(min_val, max_val, size=(a_0.size,), dtype=dtype)  # P2 -> P0
    delta_1 = sub_mode_L_minus_one(delta, delta_0)
    eta_p = eta_pp ^ (x > (r - 1))

    eta_p_0 = rng.integers(min_val, max_val, size=(a_0.size,), dtype=dtype)  # P2 -> P0
    eta_p_1 = sub_mode_L_minus_one(eta_p, eta_p_0)  # P2 -> P1

    t0 = eta_pp * eta_p_0
    t1 = add_mode_L_minus_one(t0, t0)
    t2 = sub_mode_L_minus_one(eta_pp, t1)
    eta_0 = add_mode_L_minus_one(eta_p_0, t2)

    t0 = eta_pp * eta_p_1
    t1 = add_mode_L_minus_one(t0, t0)
    eta_1 = sub_mode_L_minus_one(eta_p_1, t1)

    t0 = add_mode_L_minus_one(delta_0, eta_0)
    t1 = sub_mode_L_minus_one(t0, dtype(1))
    t2 = sub_mode_L_minus_one(t1, alpha)
    theta_0 = add_mode_L_minus_one(beta_0, t2)

    t0 = add_mode_L_minus_one(delta_1, eta_1)
    theta_1 = add_mode_L_minus_one(beta_1, t0)

    y_0 = sub_mode_L_minus_one(a_0, theta_0)
    y_1 = sub_mode_L_minus_one(a_1, theta_1)

    assert (add_mode_L_minus_one(y_0, y_1) == (a_0 + a_1)).all()
