from research.secure_inference_3pc.backend import backend
from research.secure_inference_3pc.modules.base import SecureModule
from research.secure_inference_3pc.conv2d.conv2d_handler_factory import conv2d_handler_factory
from research.secure_inference_3pc.modules.maxpool import SecureMaxPool
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, MIN_VAL, MAX_VAL, SIGNED_DTYPE, \
    COMPARISON_NUM_BITS_IGNORED, TRUNC_BITS, P, NUM_BITS
from research.secure_inference_3pc.modules.bReLU import SecureOptimizedBlockReLU, post_brelu
from research.secure_inference_3pc.parties.client.numba_methods import private_compare_numba, post_compare_numba, \
    mult_client_numba


class SecureConv2DClient(SecureModule):

    def __init__(self, W_shape, stride, dilation, padding, groups, **kwargs):
        super(SecureConv2DClient, self).__init__(**kwargs)

        self.W_shape = W_shape
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups
        self.conv2d_handler = conv2d_handler_factory.create(self.device)

    def forward(self, X_share):
        assert self.W_shape[2] == self.W_shape[3]
        assert (self.W_shape[1] == X_share.shape[1]) or self.groups > 1

        W_share = self.prf_handler[CLIENT, SERVER].integers(low=MIN_VAL, high=MAX_VAL, size=self.W_shape, dtype=SIGNED_DTYPE)
        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=W_share.shape, dtype=SIGNED_DTYPE)

        E_share = backend.subtract(X_share, A_share, out=A_share)
        F_share = backend.subtract(W_share, B_share, out=B_share)

        self.network_assets.sender_01.put(E_share)
        self.network_assets.sender_01.put(F_share)

        E_share_server = self.network_assets.receiver_01.get()
        F_share_server = self.network_assets.receiver_01.get()

        E = backend.add(E_share_server, E_share, out=E_share)
        F = backend.add(F_share_server, F_share, out=F_share)

        out = self.conv2d_handler.conv2d(X_share,
                                         F,
                                         E,
                                         W_share,
                                         padding=self.padding,
                                         stride=self.stride,
                                         dilation=self.dilation,
                                         groups=self.groups)

        C_share = self.network_assets.receiver_02.get()
        out = backend.add(out, C_share, out=out)
        out = backend.right_shift(out, TRUNC_BITS, out=out)

        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=out.shape, dtype=SIGNED_DTYPE)

        out = backend.add(out, mu_0, out=out)

        return out


class PrivateCompareClient(SecureModule):
    def __init__(self, **kwargs):
        super(PrivateCompareClient, self).__init__(**kwargs)

    def forward(self, x_bits_0, r, beta):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=backend.int8)
        d_bits_0 = private_compare_numba(s, r, x_bits_0, beta, COMPARISON_NUM_BITS_IGNORED)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)

        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, **kwargs):
        super(ShareConvertClient, self).__init__(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def forward(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        r = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=a_0.shape, dtype=SIGNED_DTYPE)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [
            NUM_BITS - COMPARISON_NUM_BITS_IGNORED], dtype=backend.int8)

        alpha = backend.astype(0 < r_0 - r, SIGNED_DTYPE)
        a_tild_0 = a_0 + r_0
        self.network_assets.sender_02.put(a_tild_0)

        beta_0 = backend.astype(0 < a_0 - a_tild_0, SIGNED_DTYPE)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)
        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL, size=a_0.shape,
                                                                     dtype=SIGNED_DTYPE)

        return post_compare_numba(a_0, eta_pp, delta_0, alpha, beta_0, mu_0, eta_p_0)


class SecureMultiplicationClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMultiplicationClient, self).__init__(**kwargs)


    def forward(self, X_share, Y_share):
        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=X_share.shape, dtype=SIGNED_DTYPE)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        C_share = self.network_assets.receiver_02.get()
        out = mult_client_numba(X_share, Y_share, C_share, mu_0, E_share, E_share_server, F_share, F_share_server)

        return out


class SecureMSBClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureMSBClient, self).__init__(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.private_compare = PrivateCompareClient(**kwargs)

    def forward(self, a_0):
        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=backend.int8)
        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [NUM_BITS - COMPARISON_NUM_BITS_IGNORED], dtype=backend.int8)
        x_bit_0_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=a_0.shape, dtype=a_0.dtype)

        x_0 = self.network_assets.receiver_02.get()
        r_1 = self.network_assets.receiver_01.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2

        self.private_compare(x_bits_0, r, beta)

        beta = backend.astype(beta, SIGNED_DTYPE)
        beta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=mu_0.shape,
                                                                      dtype=SIGNED_DTYPE)

        gamma_0 = beta_p_0 - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)

        alpha_0 = gamma_0 + delta_0 - 2 * theta_0
        alpha_0 = alpha_0 + mu_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    # counter = 0
    def __init__(self, **kwargs):
        super(SecureDReLUClient, self).__init__(**kwargs)

        self.share_convert = ShareConvertClient(**kwargs)
        self.msb = SecureMSBClient(**kwargs)

    def forward(self, X_share):
        # SecureDReLUClient.counter += 1
        # np.save("/home/yakir/Data2/secure_activation_statistics/client/{}.npy".format(SecureDReLUClient.counter), X_share)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=X_share.shape, dtype=SIGNED_DTYPE)

        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)

        return -MSB_0 + mu_0


class SecureReLUClient(SecureModule):
    def __init__(self, dummy_relu=False, **kwargs):
        super(SecureReLUClient, self).__init__(**kwargs)

        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, X_share):

        shape = X_share.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        X_share = X_share.flatten()
        MSB_0 = self.DReLU(X_share)
        ret = self.mult(X_share, MSB_0).reshape(shape)

        return ret + mu_0


class SecurePostBReLUMultClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecurePostBReLUMultClient, self).__init__(**kwargs)

    def forward(self, activation, sign_tensors, cumsum_shapes, pad_handlers, is_identity_channels, active_block_sizes,
                active_block_sizes_to_channels):
        non_identity_activation = activation[:, ~is_identity_channels]

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=non_identity_activation.shape, dtype=SIGNED_DTYPE)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(MIN_VAL, MAX_VAL + 1, size=sign_tensors.shape, dtype=SIGNED_DTYPE)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=activation.shape, dtype=SIGNED_DTYPE)

        E_share = non_identity_activation - A_share
        F_share = sign_tensors - B_share

        self.network_assets.sender_01.put(E_share)
        E_share_server = self.network_assets.receiver_01.get()

        self.network_assets.sender_01.put(F_share)
        F_share_server = self.network_assets.receiver_01.get()

        C_share = self.network_assets.receiver_02.get()
        E = E_share_server + E_share
        F = F_share_server + F_share

        F = post_brelu(activation, F, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)[:, ~is_identity_channels]
        sign_tensors = post_brelu(activation, sign_tensors, cumsum_shapes, pad_handlers, active_block_sizes, active_block_sizes_to_channels)[:, ~is_identity_channels]

        out = non_identity_activation * F + sign_tensors * E + C_share
        activation[:, ~is_identity_channels] = out

        activation = activation + mu_0

        return activation


class SecureBlockReLUClient(SecureModule, SecureOptimizedBlockReLU):
    def __init__(self, block_sizes, dummy_relu=False, **kwargs):
        SecureModule.__init__(self, **kwargs)
        SecureOptimizedBlockReLU.__init__(self, block_sizes)
        self.DReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)
        self.post_bReLU = SecurePostBReLUMultClient(**kwargs)

    def forward(self, activation):

        return SecureOptimizedBlockReLU.forward(self, activation)


class SecureSelectShareClient(SecureModule):
    def __init__(self, **kwargs):
        super(SecureSelectShareClient, self).__init__(**kwargs)
        self.secure_multiplication = SecureMultiplicationClient(**kwargs)

    def forward(self, alpha, x, y):
        # if alpha == 0: return x else return y
        shape = alpha.shape
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL + 1, size=shape, dtype=SIGNED_DTYPE)

        w = y - x
        c = self.secure_multiplication(alpha, w)
        z = x + c
        return z + mu_0


class SecureMaxPoolClient(SecureMaxPool):
    def __init__(self, kernel_size=3, stride=2, padding=1, **kwargs):
        super(SecureMaxPoolClient, self).__init__(kernel_size, stride, padding, **kwargs)
        self.select_share = SecureSelectShareClient(**kwargs)
        self.dReLU = SecureDReLUClient(**kwargs)
        self.mult = SecureMultiplicationClient(**kwargs)

    def forward(self, x):
        ret = super(SecureMaxPoolClient, self).forward(x)
        mu_0 = self.prf_handler[CLIENT, SERVER].integers(MIN_VAL, MAX_VAL, size=ret.shape, dtype=SIGNED_DTYPE)

        return ret + mu_0
