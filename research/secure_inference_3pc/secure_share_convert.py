import torch
import numpy as np
import random
from tqdm import tqdm
num_bits = 64
L = 2 ** 64

for _ in tqdm(range(1000000)):
    a_0 = int(''.join([str(x) for x in np.random.randint(0, 2, size=num_bits)]), 2)
    a_1 = int(''.join([str(x) for x in np.random.randint(0, 2, size=num_bits)]), 2)

    r_0 = int(''.join([str(x) for x in np.random.randint(0, 2, size=num_bits)]), 2)
    r_1 = int(''.join([str(x) for x in np.random.randint(0, 2, size=num_bits)]), 2)

    r = (r_0 + r_1) % L
    alpha = (r_0 + r_1) > L

    eta_pp = np.random.randint(0, 2)

    a_tild_0 = (a_0 + r_0) % L
    beta_0 = (a_0 + r_0) > L

    a_tild_1 = (a_1 + r_1) % L
    beta_1 = (a_1 + r_1) > L

    x = (a_tild_0 + a_tild_1) % L
    delta = (a_tild_0 + a_tild_1) > L
    delta_0 = random.randint(0, L - 1)
    delta_1 = (delta - delta_0) % (L - 1)

    eta_p = eta_pp ^ (x > (r - 1))

    eta_p_0 = random.randint(0, L - 1)
    eta_p_1 = (eta_p - eta_p_0) % (L - 1)

    eta_0 = eta_p_0 + (1 - 0) * eta_pp - 2 * eta_pp * eta_p_0
    eta_1 = eta_p_1 + (1 - 1) * eta_pp - 2 * eta_pp * eta_p_1

    theta_0 = beta_0 + (1 - 0) * (-alpha - 1) + delta_0 + eta_0
    theta_1 = beta_1 + (1 - 1) * (-alpha - 1) + delta_1 + eta_1

    y_0 = a_0 - theta_0
    y_1 = a_1 - theta_1

    assert (y_0 + y_1) % (L-1) == (a_0 + a_1) % L

