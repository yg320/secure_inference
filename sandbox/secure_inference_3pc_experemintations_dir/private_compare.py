import numpy as np
from tqdm import tqdm
dtype = np.uint64
powers = np.arange(64, dtype=dtype)[np.newaxis]
moduli = (2 ** powers)
p = dtype(np.uint8(67))
min_val = np.iinfo(dtype).min
max_val = np.iinfo(dtype).max

def sub_mode_p(x, y):
    mask = y > x
    ret = x - y
    ret_2 = x + (p - y)
    ret[mask] = ret_2[mask]
    return ret


def decompose(value):
    orig_shape = list(value.shape)
    value_bits = (value.reshape(-1, 1) & moduli) >> powers
    return value_bits.reshape(orig_shape + [64])

for seed in tqdm(range(300)):
    prf_0 = np.random.default_rng(seed=seed)
    prf_1 = np.random.default_rng(seed=seed)
    prf_2 = np.random.default_rng(seed=seed)

    x = prf_0.integers(min_val, max_val + 1, size=(1,10, 20, 20), dtype=dtype)
    _ = prf_1.integers(min_val, max_val + 1, size=(1,10, 20, 20), dtype=dtype)
    r = prf_0.integers(min_val, max_val + 1, size=(1,10, 20, 20), dtype=dtype)
    _ = prf_1.integers(min_val, max_val + 1, size=(1,10, 20, 20), dtype=dtype)
    r[0] = max_val
    beta = prf_0.integers(0, 2, size=(1,10, 20, 20), dtype=dtype)
    _ = prf_1.integers(0, 2, size=(1,10, 20, 20), dtype=dtype)
    x_bits = decompose(x)

    x_bits_0 = prf_0.integers(low=0, high=67, size=x_bits.shape, dtype=dtype)
    _ = prf_1.integers(low=0, high=67, size=x_bits.shape, dtype=dtype)
    x_bits_1 = sub_mode_p(x_bits, x_bits_0)





    s = prf_0.integers(low=1, high=67, size=x_bits.shape, dtype=dtype)
    _ = prf_1.integers(low=1, high=67, size=x_bits.shape, dtype=dtype)
    u = prf_0.integers(low=1, high=67, size=x_bits.shape, dtype=dtype)
    _ = prf_1.integers(low=1, high=67, size=x_bits.shape, dtype=dtype)


    t = r + dtype(1)
    r_bits = decompose(r)
    t_bits = decompose(t)

    P = 67
    def get_c_case_0(x_bits, r_bits, j):
        w = (((x_bits + j * r_bits) % P) + ((dtype(P) - ((dtype(2) * r_bits * x_bits) % P)) % P))
        rrr = sub_mode_p(w[..., ::-1].cumsum(axis=-1)[..., ::-1] % P, w)
        a = dtype((((j * r_bits + j) % P) + ((P - x_bits) % P)) % P)
        return (a + rrr) % P

    def get_c_case_1(x_bits, t_bits, j):
        a = x_bits + j * t_bits
        b = (dtype(2) * t_bits * x_bits) % p
        c = p - b
        w = (a + c) % p
        rrr = sub_mode_p(w[..., ::-1].cumsum(axis=-1)[..., ::-1] % p, w)

        f = (p - j * t_bits + x_bits) % p

        h = (f + j) % p
        a = dtype(h)

        return (a+rrr) % p

    def get_c_case_2(u, j):
        c = (p + 1 - j) * (u + 1) + (p-j) * u
        c[..., 0] = u[...,0] * (p-1) ** j
        return c % p

    c_bits_case_0_0 = get_c_case_0(x_bits_0, r_bits, dtype(0))
    c_bits_case_0_1 = get_c_case_0(x_bits_1, r_bits, dtype(1))
    c_bits_case_1_0 = get_c_case_1(x_bits_0, t_bits, dtype(0))
    c_bits_case_1_1 = get_c_case_1(x_bits_1, t_bits, dtype(1))
    c_bits_case_2_0 = get_c_case_2(u, dtype(0))
    c_bits_case_2_1 = get_c_case_2(u, dtype(1))

    c_bits_0 = c_bits_case_0_0
    c_bits_0[np.logical_and(beta == 1, r!=max_val)] = c_bits_case_1_0[np.logical_and(beta == 1, r!=max_val)]
    c_bits_0[np.logical_and(beta == 1, r==max_val)] = c_bits_case_2_0[np.logical_and(beta == 1, r==max_val)]

    c_bits_1 = c_bits_case_0_1
    c_bits_1[np.logical_and(beta == 1, r!=max_val)] = c_bits_case_1_1[np.logical_and(beta == 1, r!=max_val)]
    c_bits_1[np.logical_and(beta == 1, r==max_val)] = c_bits_case_2_1[np.logical_and(beta == 1, r==max_val)]

    d_bits_0 = (s*c_bits_0) % p
    d_bits_1 = (s*c_bits_1) % p

    # pi = np.repeat(np.arange(64)[np.newaxis], 1000, axis=0)
    # pi = prf_0.permuted(pi, axis=1)
    # for i in range(1000):
    #     d_bits_0[i] = d_bits_0[i, pi[i]]
    #     d_bits_1[i] = d_bits_1[i, pi[i]]

    d_bits_0 = prf_0.permutation(d_bits_0, axis=-1)
    d_bits_1 = prf_1.permutation(d_bits_1, axis=-1)


    d = (d_bits_0 + d_bits_1) % p
    beta_p = (d == 0).any(axis=-1).astype(dtype)

    assert np.all((beta ^ (x > r)) == beta_p), seed
