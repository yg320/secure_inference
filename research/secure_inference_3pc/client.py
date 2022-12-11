import torch
import numpy as np

from research.communication.utils import Sender, Receiver
from research.secure_inference_3pc.base import SecureModule, NetworkAssets, CryptoAssets, pre_conv, mat_mult, post_conv, Addresses, decompose, get_c, P

from research.distortion.utils import get_model
from research.pipeline.backbones.secure_resnet import AvgPoolResNet


class SecureConv2DClient(SecureModule):
    def __init__(self, W, stride, dilation, padding, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        t0 = time.time()
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert self.W_share.shape[1] == X_share.shape[1]

        A_share = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
        B_share = self.crypto_assets.get_random_tensor_over_L(shape=self.W_share.shape, prf=self.crypto_assets.prf_02_torch)

        C_share = torch.from_numpy(self.network_assets.receiver_02.get())

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        E_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
        self.network_assets.sender_01.put(E_share)
        F_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
        self.network_assets.sender_01.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share
        t1 = time.time()
        X_share = X_share.numpy()
        F = F.numpy()
        E = E.numpy()
        self.W_share = self.W_share.numpy()

        X_share, F, batch_size, nb_channels_out, nb_rows_out, nb_cols_out = pre_conv(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
        E, self.W_share, _, _, _, _ = pre_conv(E, self.W_share, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

        X_share = X_share.copy()
        F = F.copy()
        E = E.copy()
        self.W_share = self.W_share.copy()

        print(f"SecureConv2DClient extras finished - {time.time() - t1}")

        out_numpy = mat_mult(X_share[0], F, E[0], self.W_share)
        out_numpy = out_numpy[np.newaxis]
        out_numpy = post_conv(None, out_numpy, batch_size, nb_channels_out, nb_rows_out, nb_cols_out)
        out = torch.from_numpy(out_numpy)
        out = out + C_share
        # out = \
        #     torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
        #     torch.conv2d(E, self.W_share, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
        #     C_share
        out = out // self.trunc
        print(f"SecureConv2DClient finished - {time.time() - t0}")

        return out


class SecureConv2DClient_V2(SecureModule):
    def __init__(self, W, stride, dilation, padding, crypto_assets, network_assets):
        super(SecureConv2DClient, self).__init__(crypto_assets, network_assets)

        self.W_share = W
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, X_share):
        print(f"SecureConv2DClient start ({X_share.shape}, {self.W_share.shape})")
        t0 = time.time()
        assert self.W_share.shape[2] == self.W_share.shape[3]
        assert self.W_share.shape[1] == X_share.shape[1]
        # assert X_share.shape[2] == X_share.shape[3]

        A_share = self.crypto_assets.get_random_tensor_over_L(shape=X_share.shape, prf=self.crypto_assets.prf_02_torch)
        B_share = self.crypto_assets.get_random_tensor_over_L(shape=self.W_share.shape, prf=self.crypto_assets.prf_02_torch)

        C_share = torch.from_numpy(self.network_assets.receiver_02.get())

        E_share = X_share - A_share
        F_share = self.W_share - B_share

        E_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
        self.network_assets.sender_01.put(E_share)
        F_share_server = torch.from_numpy(self.network_assets.receiver_01.get())
        self.network_assets.sender_01.put(F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share
        print(f"SecureConv2DClient computation start ({X_share.shape}, {self.W_share.shape})")
        t1 = time.time()
        out = \
            torch.conv2d(X_share, F, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
            torch.conv2d(E, self.W_share, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1) + \
            C_share
        out = out // self.trunc
        print(f"SecureConv2DClient computation finished - {time.time() - t1}")
        print(f"SecureConv2DClient finished - {time.time() - t0}")

        return out


class PrivateCompareClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(PrivateCompareClient, self).__init__(crypto_assets, network_assets)

    def forward(self, x_bits_0, r, beta):

        s = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_0.shape, dtype=self.crypto_assets.numpy_dtype)
        u = self.crypto_assets.prf_01_numpy.integers(low=1, high=67, size=x_bits_0.shape, dtype=self.crypto_assets.numpy_dtype)

        t = r + self.crypto_assets.numpy_dtype(1)
        party = self.crypto_assets.numpy_dtype(0)
        r_bits = decompose(r)
        t_bits = decompose(t)

        c_bits_0 = get_c(x_bits_0, r_bits, t_bits, beta, party)

        d_bits_0 = (s * c_bits_0) % P
        d_bits_0 = self.crypto_assets.prf_01_numpy.permutation(d_bits_0, axis=-1)
        self.network_assets.sender_02.put(d_bits_0)

        return


class ShareConvertClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(ShareConvertClient, self).__init__(crypto_assets, network_assets)
        self.private_compare = PrivateCompareClient(crypto_assets, network_assets)

    def forward(self, a_0):

        eta_pp = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        r = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)
        r_0 = self.crypto_assets.prf_01_numpy.integers(self.min_val, self.max_val + 1, size=a_0.shape, dtype=self.dtype)

        alpha = (r < r_0).astype(self.dtype)

        a_tild_0 = a_0 + r_0
        beta_0 = (a_tild_0 < a_0).astype(self.dtype)
        self.network_assets.sender_02.put(a_tild_0)

        x_bits_0 = self.network_assets.receiver_02.get()
        delta_0 = self.network_assets.receiver_02.get()

        self.private_compare(x_bits_0, r - 1, eta_pp)

        eta_p_0 = self.network_assets.receiver_02.get()

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

        beta = self.crypto_assets.prf_01_numpy.integers(0, 2, size=a_0.shape, dtype=self.dtype)

        x_bits_0 = self.network_assets.receiver_02.get()
        x_0 = self.network_assets.receiver_02.get()
        x_bit_0_0 = self.network_assets.receiver_02.get()

        y_0 = self.add_mode_L_minus_one(a_0, a_0)
        r_0 = self.add_mode_L_minus_one(x_0, y_0)
        r_1 = self.network_assets.receiver_01.get()
        self.network_assets.sender_01.put(r_0)
        r = self.add_mode_L_minus_one(r_0, r_1)

        self.private_compare(x_bits_0, r, beta)
        # execute_secure_compare

        beta_p_0 = self.network_assets.receiver_02.get()

        gamma_0 = beta_p_0 + (0 * beta) - (2 * beta * beta_p_0)
        delta_0 = x_bit_0_0 + (0 * (r % 2)) - (2 * (r % 2) * x_bit_0_0)

        theta_0 = self.mult(gamma_0, delta_0)
        alpha_0 = gamma_0 + delta_0 - 2 * theta_0

        return alpha_0


class SecureDReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureDReLUClient, self).__init__(crypto_assets, network_assets)

        self.share_convert = ShareConvertClient(crypto_assets, network_assets)
        self.msb = SecureMSBClient(crypto_assets, network_assets)

    def forward(self, X_share):
        X0_converted = self.share_convert(X_share)
        MSB_0 = self.msb(X0_converted)
        return -MSB_0


class SecureReLUClient(SecureModule):
    def __init__(self, crypto_assets, network_assets):
        super(SecureReLUClient, self).__init__(crypto_assets, network_assets)

        self.DReLU = SecureDReLUClient(crypto_assets, network_assets)
        self.mult = SecureMultiplicationClient(crypto_assets, network_assets)

    def forward(self, X_share):
        t0 = time.time()
        X_share = X_share.numpy().astype(self.dtype)
        MSB_0 = self.DReLU(X_share)
        relu_0 = self.mult(X_share, MSB_0)
        print(f"SecureReLUClient finished - {time.time() - t0}")

        return torch.from_numpy(relu_0.astype(self.signed_type))


def build_secure_conv(crypto_assets, network_assets, module):
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


if __name__ == "__main__":

    config_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py"
    image_path = "/home/yakir/tmp/image_0.pt"
    model_path = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/iter_80000.pth"

    addresses = Addresses()
    port_01 = addresses.port_01
    port_10 = addresses.port_10
    port_02 = addresses.port_02
    port_20 = addresses.port_20
    port_12 = addresses.port_12
    port_21 = addresses.port_21

    prf_01_seed = 0
    prf_02_seed = 1
    prf_12_seed = 2

    sender_01 = Sender(port_01)
    sender_02 = Sender(port_02)
    sender_12 = None
    receiver_01 = Receiver(port_10)
    receiver_02 = Receiver(port_20)
    receiver_12 = None

    prf_01_numpy = np.random.default_rng(seed=prf_01_seed)
    prf_02_numpy = np.random.default_rng(seed=prf_02_seed)
    prf_12_numpy = None
    prf_01_torch = torch.Generator().manual_seed(prf_01_seed)
    prf_02_torch = torch.Generator().manual_seed(prf_02_seed)
    prf_12_torch = None

    crypto_assets = CryptoAssets(
        prf_01_numpy=prf_01_numpy,
        prf_02_numpy=prf_02_numpy,
        prf_12_numpy=prf_12_numpy,
        prf_01_torch=prf_01_torch,
        prf_02_torch=prf_02_torch,
        prf_12_torch=prf_12_torch,
    )

    network_assets = NetworkAssets(
        sender_01=sender_01,
        sender_02=sender_02,
        sender_12=sender_12,
        receiver_01=receiver_01,
        receiver_02=receiver_02,
        receiver_12=receiver_12
    )

    model = get_model(
        config=config_path,
        gpu_id=None,
        checkpoint_path=None
    )

    model.backbone.stem[0] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[0])
    model.backbone.stem[1] = torch.nn.Identity()
    model.backbone.stem[2] = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[3] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[3])
    model.backbone.stem[4] = torch.nn.Identity()
    model.backbone.stem[5] = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    model.backbone.stem[6] = build_secure_conv(crypto_assets, network_assets, model.backbone.stem[6])
    model.backbone.stem[7] = torch.nn.Identity()
    model.backbone.stem[8] = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    for layer in [1, 2, 3, 4]:
        for block in [0, 1]:

            cur_res_layer = getattr(model.backbone, f"layer{layer}")
            cur_res_layer[block].conv1 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv1)
            cur_res_layer[block].bn1 = torch.nn.Identity()
            cur_res_layer[block].relu_1 = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

            cur_res_layer[block].conv2 = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].conv2)
            cur_res_layer[block].bn2 = torch.nn.Identity()
            cur_res_layer[block].relu_2 = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

            if cur_res_layer[block].downsample:
                cur_res_layer[block].downsample = build_secure_conv(crypto_assets, network_assets, cur_res_layer[block].downsample[0])


    model.decode_head.image_pool[1].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.image_pool[1].conv)
    model.decode_head.image_pool[1].bn = torch.nn.Identity()
    model.decode_head.image_pool[1].activate = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    for i in range(4):
        model.decode_head.aspp_modules[i].conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.aspp_modules[i].conv)
        model.decode_head.aspp_modules[i].bn = torch.nn.Identity()
        model.decode_head.aspp_modules[i].activate = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.bottleneck.conv = build_secure_conv(crypto_assets, network_assets, model.decode_head.bottleneck.conv)
    model.decode_head.bottleneck.bn = torch.nn.Identity()
    model.decode_head.bottleneck.activate = SecureReLUClient(crypto_assets=crypto_assets, network_assets=network_assets)

    model.decode_head.conv_seg = build_secure_conv(crypto_assets, network_assets, model.decode_head.conv_seg)
    model.decode_head.image_pool[0].forward = lambda x: x.sum(dim=[2, 3], keepdims=True) // (x.shape[2] * x.shape[3])


    I = (torch.load(image_path).unsqueeze(0) * crypto_assets.trunc).to(crypto_assets.torch_dtype)
    I1 = crypto_assets.get_random_tensor_over_L(I.shape, prf=crypto_assets.prf_01_torch)
    I0 = I - I1
    import time
    time.sleep(5)
    print("Start")
    image = I0
    t0 = time.time()
    out_0 = model.backbone.stem(image)
    out_1 = network_assets.receiver_01.get()

    out = (torch.from_numpy(out_1) + out_0)
    out = out.to(torch.float32) / crypto_assets.trunc

    model_baseline = get_model(
        config=config_path,
        gpu_id=None,
        checkpoint_path=model_path
    )
    desired_out = model_baseline.backbone.stem(torch.load(image_path).unsqueeze(0))
    print((out - desired_out).abs().max())
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