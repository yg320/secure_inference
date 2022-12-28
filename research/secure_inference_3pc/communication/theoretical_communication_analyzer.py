import torch
import research
import pickle

from research.distortion.utils import get_model, ArchUtilsFactory
from research.pipeline.backbones.secure_resnet import MyResNet, SecureBottleneck, SecureBasicBlock
from research.distortion.distortion_utils import get_brelu_bandwidth

Porthos = "Porthos"
SecureNN = "SecureNN"


class CommunicationHandler:
    def __init__(self,
                 disable_relus=False,
                 force_max_pool_instead_of_avg_pool=False,
                 include_head=True,
                 scalar_vector_optimization=False):
        self.disable_relus = disable_relus
        self.force_max_pool_instead_of_avg_pool = force_max_pool_instead_of_avg_pool
        self.include_head = include_head
        self.scalar_vector_optimization = scalar_vector_optimization
        self.l = 8

    def get_conv_cost(self, tensor_shape, conv_module):
        assert conv_module.kernel_size[0] == conv_module.kernel_size[1]
        assert conv_module.stride[0] == conv_module.stride[1]
        assert tensor_shape[1] == conv_module.in_channels, f"{tensor_shape[1]}, {conv_module.in_channels}"

        stride = conv_module.stride[0]

        groups = conv_module.groups
        m = tensor_shape[0] / stride
        i = conv_module.in_channels
        o = conv_module.out_channels
        f = conv_module.kernel_size[0]
        q = m
        # We use padding therefore q = m (and not q = m - f + 1)

        if groups == 1:
            if stride > 1:
                m_ = m * stride + f - 1
                cost_01 = m_ ** 2 * i + f ** 2 * o * i
                cost_10 = m_ ** 2 * i + f ** 2 * o * i
            else:
                cost_01 = m ** 2 * i + f ** 2 * o * i
                cost_10 = m ** 2 * i + f ** 2 * o * i
            cost_21 = m ** 2 * o
            cost = cost_01 + cost_10 + cost_21
        else:
            assert groups == o == i
            cost = groups * (2 * m ** 2 + 2 * f ** 2 + q ** 2)

        return cost * self.l, [m, o]


    def get_relu_cost(self, tensor_shape, relu_module):
        m = tensor_shape[0]
        log_p = 8

        if type(relu_module) == research.bReLU.BlockRelu:
            assert tensor_shape[1] == len(relu_module.block_sizes)
            communication_cost = sum(
                get_brelu_bandwidth(block_size=tuple(block_size), activation_dim=int(m), protocol=self.protocol, scalar_vector_optimization=self.scalar_vector_optimization, with_prf=self.with_prf) for block_size in
                relu_module.block_sizes)
        else:
            communication_cost = self.l * (((6 * log_p + 19) * m ** 2) * tensor_shape[1])

        if self.disable_relus:
            return 0, tensor_shape
        else:
            return communication_cost, tensor_shape


    def get_deeplab_decoder_cost(self, model, tensor_shape):
        cost = 0
        tensor_shape_orig = tensor_shape

        tensor_shapes = []
        cost_conv, tensor_shape = self.get_conv_cost([1, tensor_shape_orig[1]], model.decode_head.image_pool[1].conv)
        cost_relu, _ = self.get_relu_cost(tensor_shape, model.decode_head.image_pool[1].activate)
        tensor_shapes.append(tensor_shape)
        cost += (cost_conv + cost_relu)
        for module in model.decode_head.aspp_modules:
            cost_conv, tensor_shape = self.get_conv_cost(tensor_shape_orig, module.conv)
            cost_relu, _ = self.get_relu_cost(tensor_shape, module.activate)
            cost += (cost_conv + cost_relu)
            tensor_shapes.append(tensor_shape)

        tensor_shape = (tensor_shapes[0][0], sum([x[1] for x in tensor_shapes]))

        cost_conv, tensor_shape = self.get_conv_cost(tensor_shape, model.decode_head.bottleneck.conv)
        cost_relu, tensor_shape = self.get_relu_cost(tensor_shape, model.decode_head.bottleneck.activate)
        cost_conv1, tensor_shape = self.get_conv_cost(tensor_shape, model.decode_head.conv_seg)
        cost += (cost_conv + cost_relu + cost_conv1)

        return cost


    def get_resnet_block_cost(self, block, tensor_shape):
        assert type(block) == research.pipeline.backbones.secure_resnet.SecureBottleneck
        tensor_shape_orig = [tensor_shape[0], tensor_shape[1]]
        cost_conv1, tensor_shape = self.get_conv_cost(tensor_shape, block.conv1)
        cost_relu1, tensor_shape = self.get_relu_cost(tensor_shape, block.relu_1)

        cost_conv2, tensor_shape = self.get_conv_cost(tensor_shape, block.conv2)
        cost_relu2, tensor_shape = self.get_relu_cost(tensor_shape, block.relu_2)

        cost_conv3, tensor_shape = self.get_conv_cost(tensor_shape, block.conv3)

        if block.downsample:
            cost_downsample, _ = self.get_conv_cost(tensor_shape_orig, block.downsample[0])
        else:
            cost_downsample = 0
        cost_relu3, tensor_shape = self.get_relu_cost(tensor_shape, block.relu_3)

        cost = (cost_conv1 + cost_relu1 + cost_conv2 + cost_relu2 + cost_conv3 + cost_relu3 + cost_downsample)
        return cost, tensor_shape


    def get_resnet_basic_block_cost(self, block, tensor_shape):
        assert type(block) == research.pipeline.backbones.secure_resnet.SecureBasicBlock
        tensor_shape_orig = [tensor_shape[0], tensor_shape[1]]
        cost_conv1, tensor_shape = self.get_conv_cost(tensor_shape, block.conv1)
        cost_relu1, tensor_shape = self.get_relu_cost(tensor_shape, block.relu_1)

        cost_conv2, tensor_shape = self.get_conv_cost(tensor_shape, block.conv2)

        if block.downsample:
            cost_downsample, _ = self.get_conv_cost(tensor_shape_orig, block.downsample[0])
        else:
            cost_downsample = 0
        cost_relu2, tensor_shape = self.get_relu_cost(tensor_shape, block.relu_2)

        conv_cost = cost_conv1 + cost_conv2 + cost_downsample
        relu_cost = cost_relu1 + cost_relu2
        cost = conv_cost + relu_cost
        # print(relu_cost / cost)
        return cost, tensor_shape


    def get_resnet_stem_cost(self, model, tensor_shape, l=8):
        stem = model.backbone.stem
        cost_conv1, tensor_shape = self.get_conv_cost(tensor_shape, stem[0])
        cost_relu1, tensor_shape = self.get_relu_cost(tensor_shape, stem[2])
        cost_conv2, tensor_shape = self.get_conv_cost(tensor_shape, stem[3])
        cost_relu2, tensor_shape = self.get_relu_cost(tensor_shape, stem[5])
        cost_conv3, tensor_shape = self.get_conv_cost(tensor_shape, stem[6])
        cost_relu3, tensor_shape = self.get_relu_cost(tensor_shape, stem[8])

        conv_costs = cost_conv1 + cost_conv2 + cost_conv3
        relu_costs = cost_relu1 + cost_relu2 + cost_relu3
        cost = conv_costs + relu_costs
        stride = model.backbone.maxpool.stride

        if self.force_max_pool_instead_of_avg_pool or type(model.backbone.maxpool) == torch.nn.modules.pooling.MaxPool2d:
            tensor_shape[0] /= stride
            log_p = 8

            if self.protocol == Porthos:
                cost += ((model.backbone.maxpool.kernel_size - 1) * (6 * log_p + 19) * (tensor_shape[0] ** 2 * tensor_shape[1])) * self.l
            elif self.protocol == SecureNN:
                cost += ((model.backbone.maxpool.kernel_size - 1) * (8 * log_p + 24) * (tensor_shape[0] ** 2 * tensor_shape[1])) * self.l
        else:
            tensor_shape[0] /= stride
        return cost, tensor_shape


    def get_resent_cost(self, model, tensor_shape):
        cost, tensor_shape =self. get_resnet_stem_cost(model, tensor_shape=tensor_shape)

        for layer_id in range(1, 5):
            layer = getattr(model.backbone, f"layer{layer_id}")
            for block in layer:
                if type(block) == SecureBottleneck:
                    cur_cost, tensor_shape = self.get_resnet_block_cost(block, tensor_shape=tensor_shape)
                elif type(block) == SecureBasicBlock:
                    cur_cost, tensor_shape = self.get_resnet_basic_block_cost(block, tensor_shape=tensor_shape)
                else:
                    assert False
                cost += cur_cost
        if self.include_head:
            cost += self.get_deeplab_decoder_cost(model, tensor_shape)
        return cost


    def get_mobilenetv2_cost(self, model, tensor_shape):
        cost = 0

        cur_cost, tensor_shape = self.get_conv_cost(tensor_shape, model.backbone.conv1.conv)
        cost += cur_cost
        cur_cost, tensor_shape = self.get_relu_cost(tensor_shape, model.backbone.conv1.activate)
        cost += cur_cost

        for layer_id in range(1, 8):
            layer = getattr(model.backbone, f"layer{layer_id}")
            for block in layer:

                cur_cost, tensor_shape = self.get_conv_cost(tensor_shape, block.conv[0].conv)
                cost += cur_cost
                cur_cost, tensor_shape = self.get_relu_cost(tensor_shape, block.conv[0].activate)
                cost += cur_cost

                cur_cost, tensor_shape = self.get_conv_cost(tensor_shape, block.conv[1].conv)
                cost += cur_cost

                if layer_id > 1:
                    cur_cost, tensor_shape = self.get_relu_cost(tensor_shape, block.conv[1].activate)
                    cost += cur_cost

                    cur_cost, tensor_shape = self.get_conv_cost(tensor_shape, block.conv[2].conv)
                    cost += cur_cost

        if self.include_head:
            deeplab_cost = self.get_deeplab_decoder_cost(model, tensor_shape)
            cost += deeplab_cost

        return cost


config = "/home/yakir/PycharmProjects/secure_inference/work_dirs/ADE_20K/resnet_18/steps_80k/baseline_192x192_2x16/baseline_192x192_2x16.py"
baseline = get_model(
    config=config,
    gpu_id=None,
    checkpoint_path=None
)
model = get_model(
    config=config,
    gpu_id=None,
    checkpoint_path=None
)
tensor_shape = [224, 3]

relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_192x192/ResNet18/block_size_spec_0.15.pickle"
block_sizes = pickle.load(open(relu_spec_file, 'rb'))
arch_utils = ArchUtilsFactory()("AvgPoolResNet")
arch_utils.set_bReLU_layers(model, block_sizes)

# print("Porthos, PRF, YesRelMax",  CommunicationHandler(protocol=Porthos, with_prf=True, disable_relus=True, force_max_pool_instead_of_avg_pool=False, include_head=True, unoptimized_prothos=True).get_resent_cost(model, tensor_shape) / 1000000)
print(CommunicationHandler(protocol=Porthos, with_prf=True, disable_relus=False,
                           force_max_pool_instead_of_avg_pool=True, include_head=True,
                           unoptimized_prothos=False, scalar_vector_optimization=False).get_resent_cost(model,
                                                                                                       tensor_shape) / 1000000)
