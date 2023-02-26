from research.secure_inference_3pc.const import NUM_BITS
from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32


@njit((int8[:, :])(int8[:, :], NUMBA_INT_DTYPE[:], int8[:, :], int8[:], uint8), parallel=True, nogil=True,
      cache=True)
def private_compare_numba(s, r, x_bits_0, beta, ignore_bits):
    for i in prange(x_bits_0.shape[0]):
        r[i] = r[i] + beta[i]

        counter = 0

        for j in range(64 - ignore_bits):
            decompose_bit = (r[i] >> (64 - 1 - j)) & 1
            decompose_bit = -2 * decompose_bit * x_bits_0[i, j] + x_bits_0[i, j]
            counter = counter + decompose_bit

            tmp = (counter - decompose_bit) + x_bits_0[i, j] * (2 * beta[i] - 1)
            s[i, j] = (tmp * s[i, j]) % 67

    return s


@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], int8[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:],
                           NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]), parallel=True, nogil=True, cache=True)
def post_compare_numba(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0):
    out = a_0
    for i in prange(a_0.shape[0], ):

        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])
        t0 = eta_pp_t * eta_p_0[i]

        # t1 = add_mode_L_minus_one(t0, t0)
        t1 = t0 + t0
        if NUMBA_UINT_DTYPE(t0) > NUMBA_UINT_DTYPE(t1):
            t1 += 1
        if t1 == -1:
            t1 = 0

        # t2 = sub_mode_L_minus_one(eta_pp_t, t1)
        if NUMBA_UINT_DTYPE(t1) > NUMBA_UINT_DTYPE(eta_pp_t):
            t2 = eta_pp_t - t1 - 1
        else:
            t2 = eta_pp_t - t1

        # eta_0 = add_mode_L_minus_one(eta_p_0[i], t2)
        eta_0 = eta_p_0[i] + t2
        if NUMBA_UINT_DTYPE(eta_p_0[i]) > NUMBA_UINT_DTYPE(eta_0):
            eta_0 += 1
        if eta_0 == -1:
            eta_0 = 0

        # t0 = add_mode_L_minus_one(delta_0[i], eta_0)
        t0 = delta_0[i] + eta_0
        if NUMBA_UINT_DTYPE(delta_0[i]) > NUMBA_UINT_DTYPE(t0):
            t0 += 1
        if t0 == -1:
            t0 = 0

        # t1 = sub_mode_L_minus_one(t0, 1)
        if NUMBA_UINT_DTYPE(1) > NUMBA_UINT_DTYPE(t0):
            t1 = t0 - 1 - 1
        else:
            t1 = t0 - 1

        # t2 = sub_mode_L_minus_one(t1, alpha[i])
        if NUMBA_UINT_DTYPE(alpha[i]) > NUMBA_UINT_DTYPE(t1):
            t2 = t1 - alpha[i] - 1
        else:
            t2 = t1 - alpha[i]

        # theta_0 = add_mode_L_minus_one(beta_0[i], t2)
        theta_0 = beta_0[i] + t2
        if NUMBA_UINT_DTYPE(beta_0[i]) > NUMBA_UINT_DTYPE(theta_0):
            theta_0 += 1
        if theta_0 == -1:
            theta_0 = 0

        # y_0 = sub_mode_L_minus_one(a_0, theta_0)
        if NUMBA_UINT_DTYPE(theta_0) > NUMBA_UINT_DTYPE(a_0[i]):
            y_0 = a_0[i] - theta_0 - 1
        else:
            y_0 = a_0[i] - theta_0

        # y_0 = add_mode_L_minus_one(y_0, mu_0[i])
        ret = y_0 + mu_0[i]
        if NUMBA_UINT_DTYPE(y_0) > NUMBA_UINT_DTYPE(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out


@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:],
                           NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]),
      parallel=True, nogil=True, cache=True)
def mult_client_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0], ):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


@njit(
    (NUMBA_INT_DTYPE[:, :, :, :])(NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :],
                                  NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :],
                                  NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :]), parallel=True, nogil=True,
    cache=True)
def mult_client_non_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0], ):
        f1[i] = (x[i] * (f0[i] + f1[i]) + y[i] * (e0[i] + e1[i]) + c[i]) + m[i]
    return f1


def mult_client_numba(x, y, c, m, e0, e1, f0, f1):
    if x.ndim == 1:
        return mult_client_flatten(x, y, c, m, e0, e1, f0, f1)
    else:
        return mult_client_non_flatten(x, y, c, m, e0, e1, f0, f1)
