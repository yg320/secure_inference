
import numpy as np
import time
from numba import njit, prange, uint64, int64


def add_mode_L_minus_one( a, b):
    ret = a + b
    ret[a.astype(np.uint64, copy=False) > ret.astype(np.uint64, copy=False)] += 1
    ret[ret == - 1] = 0
    return ret


def sub_mode_L_minus_one(a, b):
    ret = a - b
    ret[b.astype(np.uint64, copy=False) > a.astype(np.uint64, copy=False)] -= 1
    return ret


@njit('(int64[:])(int64[:], int8[:], int64[:], int64[:], int64[:], int64[:], int64[:])', parallel=True,  nogil=True, cache=True)
def aaa(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):

    out = a_0
    for i in prange(a_0.shape[0],):

        eta_pp_t = int64(eta_pp[i])
        t0 = eta_pp_t * eta_p_0[i]


        # t1 = add_mode_L_minus_one(t0, t0)
        t1 = t0 + t0
        if uint64(t0) > uint64(t1):
            t1 += 1
        if t1 == -1:
            t1 = 0


        # t2 = sub_mode_L_minus_one(eta_pp_t, t1)
        if uint64(t1) > uint64(eta_pp_t):
            t2 = eta_pp_t - t1 - 1
        else:
            t2 = eta_pp_t - t1



        # eta_0 = add_mode_L_minus_one(eta_p_0[i], t2)
        eta_0 = eta_p_0[i] + t2
        if uint64(eta_p_0[i]) > uint64(eta_0):
            eta_0 += 1
        if eta_0 == -1:
            eta_0 = 0



        # t0 = add_mode_L_minus_one(delta_0[i], eta_0)
        t0 = delta_0[i] + eta_0
        if uint64(delta_0[i]) > uint64(t0):
            t0 += 1
        if t0 == -1:
            t0 = 0




        # t1 = sub_mode_L_minus_one(t0, 1)
        if uint64(1) > uint64(t0):
            t1 = t0 - 1 - 1
        else:
            t1 = t0 - 1



        # t2 = sub_mode_L_minus_one(t1, alpha[i])
        if uint64(alpha[i]) > uint64(t1):
            t2 = t1 - alpha[i] - 1
        else:
            t2 = t1 - alpha[i]





        # theta_0 = add_mode_L_minus_one(beta_0[i], t2)
        theta_0 = beta_0[i] + t2
        if uint64(beta_0[i]) > uint64(theta_0):
            theta_0 += 1
        if theta_0 == -1:
            theta_0 = 0



        # y_0 = sub_mode_L_minus_one(a_0, theta_0)
        if uint64(theta_0) > uint64(a_0[i]):
            y_0 = a_0[i] - theta_0 - 1
        else:
            y_0 = a_0[i] - theta_0


        # y_0 = add_mode_L_minus_one(y_0, mu_0[i])
        ret = y_0 + mu_0[i]
        if uint64(y_0) > uint64(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out

a_0 = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))
eta_pp = np.random.randint(low=0, high=2, dtype=np.int8, size=(100000,))
delta_0 = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))
alpha = np.random.randint(low=0, high=2, dtype=np.int64, size=(100000,))
beta_0 = np.random.randint(low=0, high=2, dtype=np.int64, size=(100000,))
mu_0 = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))
eta_p_0 = np.random.randint(low=-9223129350866284269, high=9221000347554954030, dtype=np.int64, size=(100000,))

time0 = time.time()

eta_pp_t = eta_pp.astype(np.int64)
t0 = eta_pp_t * eta_p_0
t1 = add_mode_L_minus_one(t0, t0)
t2 = sub_mode_L_minus_one(eta_pp_t, t1)
eta_0 = add_mode_L_minus_one(eta_p_0, t2)

t0 = add_mode_L_minus_one(delta_0, eta_0)
t1 = sub_mode_L_minus_one(t0, np.ones_like(t0))
t2 = sub_mode_L_minus_one(t1, alpha)
theta_0 = add_mode_L_minus_one(beta_0, t2)

y_0 = sub_mode_L_minus_one(a_0, theta_0)
y_0 = add_mode_L_minus_one(y_0, mu_0)

time1 = time.time()
out = aaa(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)
time2 = time.time()
print(time1 - time0)
print(time2 - time1)

print((out == y_0).all())














#
#
#
#
#
#
#
#
#
#
#
# @njit('(int64[:])(int64[:], int8[:], int64[:], int64[:], int64[:], int64[:], int64[:])', parallel=True,  nogil=True, cache=True)
# def aaa(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):
#
#     out = np.zeros(shape=(a_0.shape[0],), dtype=np.int64)
#     for i in prange(a_0.shape[0],):
#
#         eta_pp_i = int64(eta_pp[i])
#         t0 = eta_pp_i * eta_p_0[i]
#         # t1 = add_mode_L_minus_one_numba(t0, t0)
#         a = t0
#         b = t0
#         ret = a + b
#         if uint64(a) > uint64(ret):
#             ret += 1
#         if ret == -1:
#             ret = 0
#         t1 = ret
#         # t2 = sub_mode_L_minus_one_numba(eta_pp_i, t1)
#         a = eta_pp_i
#         b = t1
#         ret = a - b
#         if uint64(b) > uint64(a):
#             ret -= 1
#         t2 = ret
#         # eta_0 = add_mode_L_minus_one_numba(eta_p_0[i], t2)
#         a = eta_p_0[i]
#         b = t2
#         ret = a + b
#         if uint64(a) > uint64(ret):
#             ret += 1
#         if ret == -1:
#             ret = 0
#         eta_0 = ret
#         # t0 = add_mode_L_minus_one_numba(delta_0[i], eta_0)
#         a = delta_0[i]
#         b = eta_0
#         ret = a + b
#         if uint64(a) > uint64(ret):
#             ret += 1
#         if ret == -1:
#             ret = 0
#         t0 = ret
#
#         a = t0
#         b = 1
#         ret = a - b
#         if uint64(b) > uint64(a):
#             ret -= 1
#         t1 = ret
#         # t1 = sub_mode_L_minus_one_numba(t0, 1)
#         # t2 = sub_mode_L_minus_one_numba(t1, alpha[i])
#         a = t1
#         b = alpha[i]
#         ret = a - b
#         if uint64(b) > uint64(a):
#             ret -= 1
#         t2 = ret
#         # theta_0 = add_mode_L_minus_one_numba(beta_0[i], t2)
#         a = beta_0[i]
#         b = t2
#         ret = a + b
#         if uint64(a) > uint64(ret):
#             ret += 1
#         if ret == -1:
#             ret = 0
#         theta_0 = ret
#
#         a = a_0[i]
#         b = theta_0
#         ret = a - b
#         if uint64(b) > uint64(a):
#             ret -= 1
#         y_0 = ret
#         # y_0 = sub_mode_L_minus_one_numba(a_0[i], theta_0)
#         # y_0 = add_mode_L_minus_one_numba(y_0, mu_0[i])
#         a = y_0
#         b = mu_0[i]
#         ret = a + b
#         if uint64(a) > uint64(ret):
#             ret += 1
#         if ret == -1:
#             ret = 0
#         y_0 = ret
#         out[i] = y_0
#     return out