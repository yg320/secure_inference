import numpy as np
import random
import torch

def compare(x, r, beta):
    return torch.bitwise_xor(x > r, beta).to(torch.long)

def get_min_max(L):
    if L == 2 ** 64:
        return -9223372036854775808, 9223372036854775807
    if L == 2 ** 64 - 1:
        return -9223372036854775807, 9223372036854775807
    if L == 67:
        return -33, 33

def generate_shares(secret, L):
    low, high = get_min_max(L)
    share_0 = torch.randint(low=low, high=high, size=secret.shape)
    share_1 = secret - share_0
    return share_0, share_1

def get_binary_rep(x):
    return bin(x)[2:].zfill(num_bits)

def decompose(tensor):
    """decompose a tensor into its binary representation."""
    torch_dtype = torch.int64
    n_bits = 64
    powers = torch.arange(n_bits, dtype=torch_dtype)
    for _ in range(len(tensor.shape)):
        powers = powers.unsqueeze(0)
    tensor = tensor.unsqueeze(-1)
    moduli = 2 ** powers
    tensor = torch.fmod((tensor // moduli.type_as(tensor)), 2)
    return tensor


a_0 = 124375432987
a_1 = 31243543
beta = False

num_bits = 64
L = 2 ** num_bits

low, high = get_min_max(L-1)
x = torch.randint(low=low, high=high, size=(1, ))
x_bit = decompose(x)

x_0, x_1 = generate_shares(x, L-1)
x_bit_0 = x_bit[..., 0]
x_bit_0_0, x_bit_0_1 = generate_shares(x_bit_0, L)
x_bit_0, x_bit_1 = generate_shares(x_bit, 67)

y_0 = 2 * a_0
y_1 = 2 * a_1

r_0 = y_0 + x_0
r_1 = y_1 + x_1

r = r_0 + r_1
r0 = decompose(r)[..., 0]

beta_prime = compare(x, r, beta)
beta_prime_0, beta_prime_1 = generate_shares(beta_prime, L)

gamma_0 = beta_prime_0 + (0 * beta) - (2 * beta * beta_prime_0)
gamma_1 = beta_prime_1 + (1 * beta) - (2 * beta * beta_prime_1)

# 8)
delta_0 = x_bit_0_0 + (0 * r0) - (2 * r0 * x_bit_0_0)
delta_1 = x_bit_0_1 + (1 * r0) - (2 * r0 * x_bit_0_1)

theta = (gamma_0 + gamma_1) * (delta_0 + delta_1)

print('fds')

msbs = []
lsbs = []
for i in range(1000):
    bits = [str(x) for x in np.random.randint(low=0, high=2, size=num_bits)]
    a = int(''.join(bits), 2)
    b = (2 * a) % (2**num_bits)
    a_bin = get_binary_rep(a)
    b_bin = get_binary_rep(b)
    msb = int(a_bin[0])
    lsb = int(b_bin[-1])

    msbs.append(msb)
    lsbs.append(lsb)

print(np.mean(np.array(msbs) == np.array(lsbs)))