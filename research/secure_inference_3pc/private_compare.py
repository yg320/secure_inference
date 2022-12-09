import numpy as np
from tqdm import tqdm
dtype = np.uint64
powers = np.arange(64, dtype=dtype)
moduli = 2 ** powers
p = np.uint8(67)
min_val = np.iinfo(dtype).min
max_val = np.iinfo(dtype).max

for seed in tqdm(range(300)):
    # seed = 1
    np.random.seed(seed)


    s = np.random.randint(low=1, high=67, size=(64,), dtype=dtype)
    pi = np.arange(64)
    np.random.shuffle(pi)

    # def add_mode_p(x, y):
    #     # x, y are uint8 and this summation cannot overflow
    #     return (x + y) % p
    #
    def sub_mode_p(x, y):
        mask = y > x
        ret = x - y
        ret_2 = x + (p - y)
        ret[mask] = ret_2[mask]
        return ret

    def decompose(value):
        value_bits = (value & moduli) >> powers
        return value_bits


    x = np.random.randint(min_val, max_val + 1, dtype=dtype)
    r = np.random.randint(min_val, max_val + 1, dtype=dtype)
    beta = 0 #np.random.randint(0, 1, dtype=dtype)

    x_bits = decompose(x)
    r_bits = decompose(r)
    t_bits = decompose(r)

    x_bits_0 = np.random.randint(low=0, high=67, size=(64,), dtype=dtype)
    x_bits_1 = sub_mode_p(x_bits, x_bits_0)

    def get_wi(x_bits, r_bits, j, i):
        return ((x_bits[i] + j * r_bits[i]) % p) + ((dtype(p) - ((dtype(2) * r_bits[i] * x_bits[i]) % p)) % p)

    def get_ci(x_bits, r_bits, j, i):

        a = dtype((((j * r_bits[i] + j) % p) + ((p - x_bits[i]) % p)) % p)
        l = np.array([get_wi(x_bits,r_bits,j,k) for k in range(i+1, 64)])
        b = (l.sum()) % p
        return (a+b) % p

    c_bits_0 = np.array([get_ci(x_bits_0, r_bits, dtype(0), i) for i in range(64)], dtype=dtype)
    c_bits_1 = np.array([get_ci(x_bits_1, r_bits, dtype(1), i) for i in range(64)], dtype=dtype)
    d_bits_0 = ((s*c_bits_0) % p)[pi]
    d_bits_1 = ((s*c_bits_1) % p)[pi]

    d = (d_bits_0 + d_bits_1) % p
    beta_p = (d == 0).any().astype(dtype)

    assert (beta ^ (x > r)) == beta_p, seed
