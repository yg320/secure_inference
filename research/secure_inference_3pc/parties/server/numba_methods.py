from research.secure_inference_3pc.const import NUM_BITS

from numba import njit, prange, int64, uint64, int8, uint8, int32, uint32

NUMBA_INT_DTYPE = int64 if NUM_BITS == 64 else int32
NUMBA_UINT_DTYPE = uint64 if NUM_BITS == 64 else uint32


@njit((int8[:, :])(int8[:, :], NUMBA_INT_DTYPE[:], int8[:, :], int8[:], uint8), parallel=True, nogil=True,
      cache=True)
def private_compare_numba_server(s, r, x_bits_1, beta, num_bits_ignored):
    # r[backend.astype(beta, backend.bool)] += 1
    # bits = self.decompose(r)
    # c_bits_1 = get_c_party_1(x_bits_1, bits, beta)
    # s = backend.multiply(s, c_bits_1, out=s)
    # d_bits_1 = module_67(s)
    for i in prange(x_bits_1.shape[0]):
        r[i] = r[i] + beta[i]
        counter = 0
        for j in range(64 - num_bits_ignored):
            multiplexer_bit = (r[i] >> (64 - 1 - j)) & 1

            w = -2 * multiplexer_bit * x_bits_1[i, j] + x_bits_1[i, j] + multiplexer_bit

            counter = counter + w
            w_cumsum = counter - w

            multiplexer_bit = multiplexer_bit - x_bits_1[i, j]
            multiplexer_bit = multiplexer_bit * (-2 * beta[i] + 1)
            multiplexer_bit = multiplexer_bit + 1
            w_cumsum = w_cumsum + multiplexer_bit

            s[i, j] = (s[i, j] * w_cumsum) % 67
    return s


@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], int8[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:],
                           NUMBA_INT_DTYPE[:]), parallel=True, nogil=True, cache=True)
def post_compare_numba(a_1, eta_pp, delta_1, beta_1, mu_0, eta_p_1):
    # mu_1 = backend.multiply(mu_0, -1, out=mu_0)
    # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
    # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
    # t11 = self.add_mode_L_minus_one(t00, t00)
    # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
    # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
    # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
    # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
    # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
    # return y_1
    out = a_1
    for i in prange(a_1.shape[0], ):
        # eta_pp = backend.astype(eta_pp, SIGNED_DTYPE)
        eta_pp_t = NUMBA_INT_DTYPE(eta_pp[i])

        # t00 = backend.multiply(eta_pp, eta_p_1, out=eta_pp)
        t00 = eta_pp_t * eta_p_1[i]

        # t11 = self.add_mode_L_minus_one(t00, t00)
        t11 = t00 + t00
        if NUMBA_UINT_DTYPE(t00) > NUMBA_UINT_DTYPE(t11):
            t11 += 1
        if t11 == -1:
            t11 = 0

        # eta_1 = self.sub_mode_L_minus_one(eta_p_1, t11)
        if NUMBA_UINT_DTYPE(t11) > NUMBA_UINT_DTYPE(eta_p_1[i]):
            eta_1 = eta_p_1[i] - t11 - 1
        else:
            eta_1 = eta_p_1[i] - t11

        # t00 = self.add_mode_L_minus_one(delta_1, eta_1)
        t00 = delta_1[i] + eta_1
        if NUMBA_UINT_DTYPE(delta_1[i]) > NUMBA_UINT_DTYPE(t00):
            t00 += 1
        if t00 == -1:
            t00 = 0

        # theta_1 = self.add_mode_L_minus_one(beta_1, t00)
        theta_1 = beta_1[i] + t00
        if NUMBA_UINT_DTYPE(t00) > NUMBA_UINT_DTYPE(theta_1):
            theta_1 += 1
        if theta_1 == -1:
            theta_1 = 0

        # y_1 = self.sub_mode_L_minus_one(a_1, theta_1)
        if NUMBA_UINT_DTYPE(theta_1) > NUMBA_UINT_DTYPE(a_1[i]):
            y_1 = a_1[i] - theta_1 - 1
        else:
            y_1 = a_1[i] - theta_1

        # y_1 = self.add_mode_L_minus_one(y_1, mu_1)
        ret = y_1 - mu_0[i]
        if NUMBA_UINT_DTYPE(y_1) > NUMBA_UINT_DTYPE(ret):
            ret += 1
        if ret == -1:
            ret = 0

        out[i] = ret
    return out


@njit((NUMBA_INT_DTYPE[:])(NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:],
                           NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:], NUMBA_INT_DTYPE[:]),
      parallel=True, nogil=True, cache=True)
def mult_server_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0], ):
        e = (e0[i] + e1[i])
        f = (f0[i] + f1[i])
        f1[i] = - e * f + x[i] * f + y[i] * e + c[i] - m[i]
    return f1


@njit(
    (NUMBA_INT_DTYPE[:, :, :, :])(NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :],
                                  NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :],
                                  NUMBA_INT_DTYPE[:, :, :, :], NUMBA_INT_DTYPE[:, :, :, :]), parallel=True, nogil=True,
    cache=True)
def mult_server_non_flatten(x, y, c, m, e0, e1, f0, f1):
    for i in prange(x.shape[0], ):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(x.shape[3]):
                    e = (e0[i, j, k, l] + e1[i, j, k, l])
                    f = (f0[i, j, k, l] + f1[i, j, k, l])
                    f1[i, j, k, l] = - e * f + x[i, j, k, l] * f + y[i, j, k, l] * e + c[i, j, k, l] - m[i, j, k, l]
        # e = (e0[i] + e1[i])
        # f = (f0[i] + f1[i])
        # f1[i] = - e * f + x[i] * f + y[i] * e + c[i] - m[i]
    return f1


def mult_server_numba(x, y, c, m, e0, e1, f0, f1):
    # E = backend.add(E_share_client, E_share, out=E_share)
    # F = backend.add(F_share_client, F_share, out=F_share)
    # out = - E * F + X_share * F + Y_share * E + C_share
    # mu_1 = backend.multiply(mu_1, -1, out=mu_1)
    # out = out + mu_1
    if x.ndim == 1:
        return mult_server_flatten(x, y, c, m, e0, e1, f0, f1)
    else:
        return mult_server_non_flatten(x, y, c, m, e0, e1, f0, f1)
