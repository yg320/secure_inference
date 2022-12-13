import torch
import numpy as np

from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, pre_conv, mat_mult, post_conv, Addresses, decompose, get_c, P, module_67, DepthToSpace, SpaceToDepth

from research.distortion.utils import get_model
from research.pipeline.backbones.secure_resnet import AvgPoolResNet
from research.pipeline.backbones.secure_aspphead import SecureASPPHead
import time

class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W.numpy()
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):

        X_share = X_share.numpy()
        assert X_share.dtype == self.signed_type
        t0 = time.time()
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert self.W_share.shape[1] == X_share.shape[1]

        A_share = self.crypto_assets.prf_02_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=X_share.shape, dtype=np.int64)
        B_share = self.crypto_assets.prf_02_numpy.integers(np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=self.W_share.shape, dtype=np.int64)
        C_share = self.network_assets.receiver_02.get()

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        E_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(E_share)
        F_share_server = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        X_share, F, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        E, self.W_share, _, _, _, _ = pre_conv(E, self.W_share, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        out_numpy = mat_mult(X_share[0], F, E[0], self.W_share)
        out_numpy = out_numpy[np.newaxis]
        out_numpy = post_conv(None, out_numpy, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        out_numpy = out_numpy + C_share

        out = out_numpy // self.trunc
        print(f"SecureConv2DClient finished - {time.time() - t0}")

        return torch.from_numpy(out)


# class SecureConv2DClient_V2(SecureModule):
#     def __init__(self, W, stride, dilation, padding, crypto_assets, network_assets):
#         super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)
#
#         self.W_share = W
#         self.stride = stride
#         self.dilation = dilation
#         self.padding = padding
#
#     def forward(self, X_share):
#         print(f"SecureConv2DClient start ({X_share.shape}, {self.W_share.shape})")
#         t0 = time.time()
#         assert self.W_share.shape[2] == self.W_share.shape[3]
#         assert self.W_share.shape[1] == X_share.shape[1]
#         # assert X_share.shape[2] == X_share.shape[3]
#
#         A_share = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
#         B_share = self.crypto_assets.get_random_tensor_over_L(shape=self.W_share.shape, prf=self.crypto_assets.prf_02_torch)
#
#         C_share = torch.from_numpy(self.network_assets.receiver_02.get())
#
#         E_share = X_share - A_share
#         F_share = self.W_share - B_share
#
#         E_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
#         self.network_assets.sender_01.put(E_share)
#         F_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
#         self.network_assets.sender_01.put(F_share)
#
#         E = E_share_server + E_share
#         F = F_share_server + F_share
#         print(f"SecureConv2DClient computation start ({X_share.shape}, {self.W_share.shape})")
#         t1 = time.time()
#         out = \
#             torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
#             torch.conv2d(E, self.W_share, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
#             C_share
#         out = out // self.trunc
#         print(f"SecureConv2DClient computation finished - {time.time() - t1}")
#         print(f"SecureConv2DClient finished - {time.time() - t0}")
#
#         return out
#

class PrivateCompareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareClient, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0, r, beta):

        if np.any(r == np.iinfo(r.dtype).max):
            assert False
        s = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_0.shape, dtype=np.int32)
        # u = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_0.shape, dtype=self.crypto_assets.numpy_dtype)
        r[beta] += 1
        bits = decompose(r)

        c_bits_0 = get_c(x_bits_0, bits, beta, np.int8(0))
        np.multiply(s, c_bits_0, out=s)
        d_bits_0 = module_67(s)

        d_bits_0 = self.crypto_assets.prf_01_numpy.permutation(d_bits_0, axis=-1)
        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertClient, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):

        eta_pp = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_0.shape, dtype=np.int8)

        r = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.crypto_assets.prf_02_numpy.integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        eta_p_0 = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val, size=a_0.shape, dtype=self.dtype)
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
        assert X_share.dtype == self.dtype
        assert Y_share.dtype == self.dtype

        cur_time = time.time()

        A_share = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
        B_share = self.crypto_assets.prf_02_numpy.integers(self.min_val, self.max_val + 1, size=X_share.shape, dtype=self.dtype)
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
        cur_time = time.time()

        beta = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_0.shape, dtype=np.int8)

        x_bits_0 = self.crypto_assets.prf_02_numpy.integers(0, P, size=list(a_0.shape) + [64], dtype=np.int8)
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
        t0 = time.time()
        shape = X_share.shape
        X_share = X_share.numpy()
        X_share = X_share.astype(self.dtype).flatten()
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0).reshape(shape)
        print(f"SecureReLUClient finished - {time.time() - t0}")
        ret = relu_0.astype(self.signed_type)
        return torch.from_numpy(ret)

from research.secure_inference_3pc.secure_block_relu import BlockReLU_V1
from research.bReLU import BlockRelu

class SecureBlockReLUClient(SecureModule):

    def __init__(self, crypto_assets, network_assets, block_sizes):
        super(SecureBlockReLUClient, self).__init__(crypto_assets, network_assets)
        self.block_sizes = np.array(block_sizes)
        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

        self.active_block_sizes = [block_size for block_size in np.unique(self.block_sizes, axis=0) if 0 not in block_size]
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
            sign_tensor = sign_tensors[int(cumsum_shapes[i]):int(cumsum_shapes[i+1])].reshape(orig_shapes[i])
            relu_map[:, channels[i]] = DepthToSpace(self.active_block_sizes[i])(sign_tensor.repeat(reshaped_inputs[i].shape[-1], axis=-1))

        activation[:, ~self.is_identity_channels] = self.mult(relu_map[:, ~self.is_identity_channels], activation[:, ~self.is_identity_channels])
        activation = activation.astype(self.signed_type)
        # real_out = network_assets.receiver_01.get() + activation
        # assert np.all(real_out == desired_out.numpy())
        return torch.from_numpy(activation)





def build_secure_conv(crypto_assets, network_assets, module, bn_module):
    return SecureConv2DClient(
        W=crypto_assets.get_random_tensor_over_L(
            shape=module.weight.shape,
            prf=crypto_assets.prf_01_torch),
        stride=module.stride,
        dilation=module.dilation,
        padding=module.padding,
        crypto_assets=crypto_assets,
        network_assets=network_assets
    )

def build_secure_relu(crypto_assets, network_assets):
    return SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

def run_inference(model, image_path, crypto_assets, network_assets):

    I = (torch.load(image_path).unsqueeze(0) * crypto_assets.trunc).to(crypto_assets.torch_dtype)[:,:,:192,:192]
    I1 = crypto_assets.get_random_tensor_over_L(I.shape, prf=crypto_assets.prf_01_torch)
    I0 = I - I1

    import time
    time.sleep(5)
    print("Start")
    image = I0
    t0 = time.time()
    out_0 = model.decode_head(model.backbone(image))
    out_1 = network_assets.receiver_01.get()
    print(time.time() - t0)
    out = (torch.from_numpy(out_1) + out_0)
    out = out.to(torch.float32) / crypto_assets.trunc
    return out

if __name__ == "__main__":
    from research.secure_inference_3pc.resnet_converter import securify_model

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py"
    secure_config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16_secure.py"
    image_path = "/home/yakir/tmp/image_0.pt"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/knapsack_0.15_192x192_2x16_finetune_80k_v2/iter_80000.pth"
    relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_192x192/ResNet18/block_size_spec_0.15.pickle"

    addresses = Addresses()

    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2

    crypto_assets = CryptoAssets(
        prf_01_numpy=np.random.default_rng(seed=prf_01_seed),
        prf_02_numpy=np.random.default_rng(seed=prf_02_seed),
        prf_12_numpy=None,
        prf_01_torch=torch.Generator().manual_seed(prf_01_seed),
        prf_02_torch=torch.Generator().manual_seed(prf_02_seed),
        prf_12_torch=None,
    )

    network_assets = NetworkAssets(
        sender_01=Sender(addresses.port_01),
        sender_02=Sender(addresses.port_02),
        sender_12=None,
        receiver_01=Receiver(addresses.port_10),
        receiver_02=Receiver(addresses.port_20),
        receiver_12=None
    )

    model = get_model(
        config=secure_config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    securify_model(model, build_secure_conv, build_secure_relu, crypto_assets, network_assets)

    import pickle
    from research.distortion.utils import ArchUtilsFactory
    from functools import partial

    SecureBlockReLUClient_partial = partial(SecureBlockReLUClient, crypto_assets=crypto_assets, network_assets=network_assets)
    layer_name_to_block_sizes = pickle.load(open(relu_spec_file, 'rb'))
    arch_utils = ArchUtilsFactory()('AvgPoolResNet')
    arch_utils.set_bReLU_layers(model, layer_name_to_block_sizes, block_relu_class=SecureBlockReLUClient_partial)

    out = run_inference(model, image_path, crypto_assets, network_assets)

    from research.bReLU import BlockRelu
    model_baseline = get_model(config=config_path, gpu_id=None, checkpoint_path=model_path)
    arch_utils.set_bReLU_layers(model_baseline, layer_name_to_block_sizes, block_relu_class=BlockRelu)


    # model_baseline.backbone.stem[2] = BlockReLU_V1(block_sizes['stem_2'])
    # model_baseline.backbone.stem[5] = BlockReLU_V1(block_sizes['stem_5'])
    # model_baseline.backbone.stem[8] = BlockReLU_V1(block_sizes['stem_8'])
    im = torch.load(image_path).unsqueeze(0)[:,:,:192,:192]
    desired_out = model_baseline.decode_head(model_baseline.backbone(im))
    print('fds')
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt
    print(np.abs((out - desired_out.detach()).numpy()).max())
    assert False

   # sudo apt-get update
   # curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
   # sudo apt-get install bzip2
   # bash Anaconda3-2019.03-Linux-x86_64.sh
   # conda create -n open-mmlab python=3.7 -y
   # exit
   # conda create -n open-mmlab python=3.7 -y
   # conda activate open-mmlab
   # conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
   # pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
   # pip install mmsegmentation
   # sudo apt-get install ffmpeg libsm6 libxext6  -y
   # conda install numba
   # https://stackoverflow.com/questions/62436205/connecting-aws-ec2-instance-using-python-socket