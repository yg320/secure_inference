import numpy as np
import torch

from research.secure_inference_3pc.modules.base import PRFFetcherModule, SecureModule
from research.secure_inference_3pc.modules.conv2d import get_output_shape, conv_2d
from research.secure_inference_3pc.const import CLIENT, SERVER, CRYPTO_PROVIDER, P
from research.secure_inference_3pc.timer import Timer
from research.secure_inference_3pc.base import decompose, get_c, module_67, TypeConverter

# TODO: change everything from dummy_tensors to dummy_tensor_shape - there is no need to pass dummy_tensors
class PRFFetcherConv2D(PRFFetcherModule):
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(PRFFetcherConv2D, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        X_share = X_share.numpy()
        out_shape = get_output_shape(X_share, self.W_share, self.padding, self.dilation, self.stride)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)

        return torch.from_numpy(np.zeros(shape=out_shape, dtype=X_share.dtype))


class PRFFetcherPrivateCompare(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherPrivateCompare, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0):
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=1, high=P, size=[x_bits_0.shape[0]] + [64], dtype=np.int32)


class PRFFetcherShareConvert(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherShareConvert, self).__init__(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):
        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, SERVER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [64], dtype=np.int8)

        self.private_compare(dummy_tensor)

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val, size=dummy_tensor.shape, dtype=self.dtype)

        return dummy_tensor


class PRFFetcherMultiplication(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMultiplication, self).__init__(crypto_assets, network_assets)


    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(self.min_val, self.max_val + 1, size=dummy_tensor.shape, dtype=self.dtype)

        return dummy_tensor


class PRFFetcherMSB(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherMSB, self).__init__(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)
        self.private_compare = PRFFetcherPrivateCompare(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.prf_handler[CLIENT, SERVER].integers_fetch(0, 2, size=dummy_tensor.shape, dtype=np.int8)
        self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers_fetch(0, P, size=list(dummy_tensor.shape) + [64], dtype=np.int8)

        self.private_compare(dummy_tensor)
        self.mult(dummy_tensor)

        return dummy_tensor


class PRFFetcherDReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherDReLU, self).__init__(crypto_assets, network_assets)

        self.share_convert = PRFFetcherShareConvert(crypto_assets, network_assets)
        self.msb = PRFFetcherMSB(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        self.share_convert(dummy_tensor)
        self.msb(dummy_tensor)

        return dummy_tensor


class PRFFetcherReLU(PRFFetcherModule):
    def __init__(self, crypto_assets, network_assets):
        super(PRFFetcherReLU, self).__init__(crypto_assets, network_assets)

        self.DReLU = PRFFetcherDReLU(crypto_assets, network_assets)
        self.mult = PRFFetcherMultiplication(crypto_assets, network_assets)

    def forward(self, dummy_tensor):

        dummy_arr = dummy_tensor.numpy().astype(self.dtype).flatten()
        self.DReLU(dummy_arr)
        self.mult(dummy_arr)
        return dummy_tensor



class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, groups, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.groups = groups

    def forward_(self, X_share):

        X_share = X_share.numpy()
        assert X_share.dtype == self.signed_type
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert (self.W_share.shape[1] == X_share.shape[1]) or self.groups > 1

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                                           size=X_share.shape, dtype=np.int64)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                                           size=self.W_share.shape, dtype=np.int64)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(np.concatenate([E_share.flatten(), F_share.flatten()]))
        E_share_server, F_share_server = \
            share_server[:E_share.size].reshape(E_share.shape), share_server[E_share.size:].reshape(F_share.shape)

        E = E_share_server + E_share
        F = F_share_server + F_share

        out_numpy = conv_2d(X_share, F, E, self.W_share, self.padding, self.stride, self.dilation, self.groups)

        out_numpy = out_numpy + C_share

        out = out_numpy // self.trunc

        return torch.from_numpy(out)

    def forward(self, X_share):
        with Timer("SecureConv2DClient"):
            return self.forward_(X_share)


class PrivateCompareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareClient, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0, r, beta):
        # with Timer("PrivateCompareClient"):
        return self.forward_(x_bits_0, r, beta)

    def forward_(self, x_bits_0, r, beta):
        if np.any(r == np.iinfo(r.dtype).max):
            assert False
        # with Timer("PrivateCompareClient - Random"):
        s = self.prf_handler[CLIENT, SERVER].integers(low=1, high=P, size=x_bits_0.shape, dtype=np.int32)
        # u = self.prf_handler[CLIENT, SERVER].integers(low=1, high=67, size=x_bits_0.shape, dtype=self.crypto_assets.numpy_dtype)
        r[beta] += 1
        bits = decompose(r)

        c_bits_0 = get_c(x_bits_0, bits, beta, np.int8(0))
        np.multiply(s, c_bits_0, out=s)
        d_bits_0 = module_67(s)

        d_bits_0 = self.prf_handler[CLIENT, SERVER].permutation(d_bits_0, axis=-1)
        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertClient, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        # with Timer("ShareConvertClient"):
        return self.forward_(a_0)
    #TODO: should be like :@Timer("ShareConvertClient")
    def forward_(self, a_0):
        eta_pp = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        r = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.prf_handler[CLIENT, SERVER].integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        eta_p_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val, size=a_0.shape, dtype=self.dtype)
        eta_pp = eta_pp.astype(self.dtype)
        t0 = eta_pp * eta_p_0
        t1 = self.add_mode_L_minus_one(t0, t0)
        t2 = self.sub_mode_L_minus_one(eta_pp, t1)
        eta_0 = self.add_mode_L_minus_one(eta_p_0, t2)

        t0 = self.add_mode_L_minus_one(delta_0, eta_0)
        t1 = self.sub_mode_L_minus_one(t0, self.dtype(1))
        t2 = self.sub_mode_L_minus_one(t1, alpha)
        theta_0 = self.add_mode_L_minus_one(beta_0, t2)

        y_0 = self.sub_mode_L_minus_one(a_0, theta_0)

        return y_0


class SecureMultiplicationClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMultiplicationClient, self).__init__(crypto_assets, network_assets)

    def forward(self, X_share, Y_share):
        # with Timer("SecureMultiplicationClient"):
        return self.forward_(X_share, Y_share)

    def forward_(self, X_share, Y_share):
        assert X_share.dtype == self.dtype
        assert Y_share.dtype == self.dtype

        A_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape,
                                                           dtype=self.dtype)
        B_share = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(self.min_val, self.max_val + 1, size=X_share.shape,
                                                           dtype=self.dtype)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(E_share)
        F_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        out = X_share * F + Y_share * E + C_share

        return out


class SecureMSBClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureMSBClient, self).__init__(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):
        # with Timer("SecureMSBClient"):
        return self.forward_(a_0)

    def forward_(self, a_0):

        beta = self.prf_handler[CLIENT, SERVER].integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.prf_handler[CLIENT, CRYPTO_PROVIDER].integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        r_mode_2 = r % 2
        self.private_compare(x_bits_0, r, beta)
        # execute_secure_compare
        beta = beta.astype(self.dtype)
        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 - (2 * r_mode_2 * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUClient, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertClient(crypto_assets, network_assets)
        self.msb = SecureMSBClient(crypto_assets, network_assets)

    def forward(self, X_share):
        assert X_share.dtype == self.dtype
        X0_converted = self.share_convert(self.dtype(2) * X_share)
        MSB_0 = self.msb(X0_converted)
        return -MSB_0


class SecureReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUClient, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

    def forward(self, X_share):
        with Timer("SecureReLUClient"):
            return self.forward_(X_share)

    def forward_(self, X_share):
        # network_assets.sender_01.put(X_share)
        # return torch.zeros_like(X_share)
        shape = X_share.shape
        X_share = X_share.numpy()
        X_share = X_share.astype(self.dtype).flatten()
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0).reshape(shape)
        ret = relu_0.astype(self.signed_type)
        return torch.from_numpy(ret)


class SecureBlockReLUClient(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUClient, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if
                                   0 not in block_size]
        self.is_identity_channels = np.array([0 in block_size for block_size in self.block_sizes])

    def forward(self, activation):
        # activation_server = network_assets.receiver_01.get()

        activation = activation.numpy()
        # desired_out = BlockReLU_V1(self.block_sizes)(activation + activation_server)
        assert activation.dtype == self.signed_type
        reshaped_inputs = []
        mean_tensors = []
        channels = []
        orig_shapes = []

        for block_size in self.active_block_sizes:
            cur_channels = [bool(x) for x in np.all(self.block_sizes == block_size, axis=1)]
            cur_input = activation[:, cur_channels]
            # reshaped_input[0,1,38,31]
            reshaped_input = SpaceToDepth(block_size)(cur_input)
            assert reshaped_input.dtype == self.signed_type
            mean_tensor = np.sum(reshaped_input, axis=-1, keepdims=True)

            channels.append(cur_channels)
            reshaped_inputs.append(reshaped_input)
            orig_shapes.append(mean_tensor.shape)
            mean_tensors.append(mean_tensor.flatten())

        cumsum_shapes = [0] + list(np.cumsum([mean_tensor.shape[0] for mean_tensor in mean_tensors]))
        mean_tensors = np.concatenate(mean_tensors)
        assert mean_tensors.dtype == self.signed_type
        activation = activation.astype(self.dtype)
        sign_tensors = self.DReLU(mean_tensors.astype(self.dtype))

        relu_map = np.ones_like(activation)
        for i in range(len(self.active_block_sizes)):
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i + 1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(
                sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels],
                                                              activation[:, ~self.is_identity_channels])
        activation = activation.astype(self.signed_type)
        # real_out = network_assets.receiver_01.get() + activation
        # assert np.all(real_out == desired_out.numpy())
        return torch.from_numpy(activation)


class PRFFetcherSecureModel(SecureModule):
    def __init__(self, model,  crypto_assets, network_assets):
        super(PRFFetcherSecureModel, self).__init__( crypto_assets, network_assets)
        self.model = model

    def forward(self, img):

        dtype = np.int64
        self.prf_handler[CLIENT, SERVER].integers_fetch(low=np.iinfo(dtype).min // 2, high=np.iinfo(dtype).max // 2, dtype=dtype, size=img.shape)
        out_0 = self.model.decode_head(self.model.backbone(torch.zeros(size=img.shape, dtype=torch.int64)))


