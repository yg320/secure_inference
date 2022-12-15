import torch
import research
import pickle

from research.distortion.utils import get_model, ArchUtilsFactory
from research.pipeline.backbones.secure_resnet import MyResNet, SecureBottleneck, SecureBasicBlock
from research.distortion.distortion_utils import get_brelu_bandwidth

Porthos = "Porthos"
SecureNN = "SecureNN"

class CommunicationHandler:
    def __init__(self, protocol=Porthos, with_prf=True, disable_relus=False,
                 force_max_pool_instead_of_avg_pool=False, include_head=True, unoptimized_prothos=False, scalar_vector_optimization=False):
        self.protocol = protocol
        self.with_prf = with_prf
        self.disable_relus = disable_relus
        self.force_max_pool_instead_of_avg_pool = force_max_pool_instead_of_avg_pool
        self.include_head = include_head
        self.unoptimized_prothos = unoptimized_prothos
        self.scalar_vector_optimization = scalar_vector_optimization
        self.l = 8

        if protocol == Porthos:
            assert with_prf, "I think Porthos default behaviour is PRF"

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
        # We use padding therefore q = m
        q = m - f + 1
        if groups == 1:
            if self.protocol == Porthos:
                assert self.with_prf
                if self.with_prf:
                    if self.unoptimized_prothos:
                        cost_01 = m ** 2 * f ** 2 * i + f ** 2 * o * i
                        cost_10 = m ** 2 * f ** 2 * i + f ** 2 * o * i
                        cost_21 = m ** 2 * o
                    else:
                        if stride > 1:
                            m_ = m * stride + f - 1
                            cost_01 = m_ ** 2 * i + f ** 2 * o * i
                            cost_10 = m_ ** 2 * i + f ** 2 * o * i
                        else:
                            cost_01 = m ** 2 * i + f ** 2 * o * i
                            cost_10 = m ** 2 * i + f ** 2 * o * i
                        cost_21 = m ** 2 * o
                    cost = cost_01 + cost_10 + cost_21

            elif self.protocol == SecureNN:
                cost = 2 * m ** 2 * f ** 2 * i + 2 * f ** 2 * o * i + m ** 2 * o
                if not self.with_prf:
                    cost *= 2
            else:
                assert False
        else:
            assert False
            assert groups == o == i
            if self.protocol == Porthos:
                cost = groups * (2 * m ** 2 + 2 * f ** 2 + q ** 2)
            else:
                assert False, "Just write the formula"

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
            if self.protocol == Porthos:
                communication_cost = self.l * (((6 * log_p + 19) * m ** 2) * tensor_shape[1])
            elif self.protocol == SecureNN:
                communication_cost = self.l * (((8 * log_p + 24) * m ** 2) * tensor_shape[1])
            else:
                assert False
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

# print("SecureNN, PRF, YesRelMax",  CommunicationHandler(protocol=SecureNN, with_prf=True, disable_relus=False, force_max_pool_instead_of_avg_pool=False).get_resent_cost(model, tensor_shape) / 1000000000)
# print("SecureNN, PRF, NoRelMax",  CommunicationHandler(protocol=SecureNN, with_prf=True, disable_relus=True, force_max_pool_instead_of_avg_pool=True).get_resent_cost(model, tensor_shape) / 1000000000)
# print("SecureNN, No PRF, YesRelMax",  CommunicationHandler(protocol=SecureNN, with_prf=False, disable_relus=False, force_max_pool_instead_of_avg_pool=False).get_resent_cost(model, tensor_shape) / 1000000000)
# print("SecureNN, No PRF, NoRelMax",  CommunicationHandler(protocol=SecureNN, with_prf=False, disable_relus=True, force_max_pool_instead_of_avg_pool=True).get_resent_cost(model, tensor_shape) / 1000000000)
#
# print("Porthos, PRF, YesRelMax",  CommunicationHandler(protocol=Porthos, with_prf=True, disable_relus=False, force_max_pool_instead_of_avg_pool=False).get_resent_cost(model, tensor_shape) / 1000000000)
# print("Porthos, PRF, NoRelMax",  CommunicationHandler(protocol=Porthos, with_prf=True, disable_relus=True, force_max_pool_instead_of_avg_pool=True).get_resent_cost(model, tensor_shape) / 1000000000)

# SecureNN, PRF, YesRelMax 3.735798448
# SecureNN, PRF, NoRelMax 1.0365836
# SecureNN, No PRF, YesRelMax 4.564764
# SecureNN, No PRF, NoRelMax 1.865549152
# Porthos, PRF, YesRelMax 2.543049392
# Porthos, PRF, NoRelMax 0.48796536

0.48796536 / 2.5430
1.865549152 / 4.564764


# SecureNN - No PRF: 4.77
# SecureNN - Yes PRF: 3.94
# SecureNN - No PRF, No ReLUs, No MaxPool: 1.65
# SecureNN - Yes PRF, No ReLUs, No MaxPool: 0.82
# Prothos - Yes PRF:

# relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_192x192/ResNet18/block_size_spec_0.15.pickle"
# block_sizes = pickle.load(open(relu_spec_file, 'rb'))

# print(ch.get_resent_cost(baseline, tensor_shape=[160, 3])/1000000000)

# arch_utils = ArchUtilsFactory()("AvgPoolResNet")
# arch_utils.set_bReLU_layers(model, block_sizes)
# print(ch.get_resent_cost(model, tensor_shape=[192, 3])/ch.get_resent_cost(baseline, tensor_shape=[192, 3]))

# print(get_resent_cost(model, tensor_shape) / get_resent_cost(baseline, tensor_shape))

# model = get_model(
#     config="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/old/mobilenet_v2_ade_20k_baseline.py",
#     gpu_id=None,
#     checkpoint_path="/home/yakir/Data2/experiments/mobilenet_v2_ade_20k/baseline_relu6/latest.pth"
# )
#
# relu_spec_file = "/home/yakir/Data2/assets_v2/deformations/ade_20k/mobilenet_v2/reduction_specs/layer_reduction_0.07.pickle"
#
#
# layer_name_to_block_size_indices = pickle.load(open(relu_spec_file, 'rb'))
# layer_name_to_block_size = {layer_name: np.array(layer_name_to_block_size_indices[layer_name][1])[layer_name_to_block_size_indices[layer_name][0]] for layer_name in layer_name_to_block_size_indices.keys()}
# arch_utils = ArchUtilsFactory()("MobileNetV2")
# arch_utils.set_bReLU_layers(model, layer_name_to_block_size)
#
# tensor_shape = [512, 3]
#
# print(get_mobilenetv2_cost(model, tensor_shape))
#
#
#
#
# 981351856.0815492/6183918464.0

#
#
#
#
# print('jeyu')
# assert False
#
#
#
# model_base = get_model(
#     config="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/m-v2_256x256_ade20k/3x4_algo.py",
#     gpu_id=None,
#     checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
# )
# model = get_model(
#     config="/home/yakir/PycharmProjects/secure_inference/research/pipeline/configs/m-v2_256x256_ade20k/3x4_algo.py",
#     gpu_id=None,
#     checkpoint_path="/home/yakir/PycharmProjects/secure_inference/work_dirs/m-v2_256x256_ade20k/baseline/iter_160000.pth"
# )
# #
# relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_0.0625/block_spec.pickle"
# relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_32.pickle"
# relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_relu_reduction_01_10_inf.pickle" # v5
# relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_relu_reduction.pickle" # v3
# relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_32.pickle" # v2
# relu_spec_file = "/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_bandwidth_div_512_only_identity_no_block_10x_for_identity.pickle" # v7
# relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity/block_spec_bandwidth_div_512_only_identity_no_block.pickle" # v6
# relu_spec_file =f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity_64/block_spec_exclude_special_blocks_false.pickle"
# relu_spec_file =f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/2_groups_160k_with_identity_64/block_spec_exclude_special_blocks_false.pickle"
# relu_spec_file =f"/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/snr/block_size_spec.pickle"
# relu_spec_file ="/home/yakir/Data2/assets_v4/distortions/ade_20k_256x256/MobileNetV2/one_group_mini_4000_channels/block_size_spec_0.pickle"
#
# layer_name_to_block_size_indices = pickle.load(open(relu_spec_file, 'rb'))
# # import numpy as np
# # layer_name_to_block_size_indices = {k: 3*np.ones_like(v) for k, v in layer_name_to_block_size_indices.items()}
# # for k in layer_name_to_block_size_indices.keys():
# #     layer_name_to_block_size_indices[k][:,1] = 4
# arch_utils = ArchUtilsFactory()("MobileNetV2")
# arch_utils.set_bReLU_layers(model, layer_name_to_block_size_indices)
# tensor_shape = [256, 3]
#
# # print(get_mobilenetv2_cost(model_base, tensor_shape))
# # 877566976/12583738368
# # print(get_mobilenetv2_cost(model, tensor_shape))
# print(get_mobilenetv2_cost(model, tensor_shape)/get_mobilenetv2_cost(model_base, tensor_shape))
# # print(get_mobilenetv2_cost(model_base, tensor_shape)/1000000000)
# # print(get_mobilenetv2_cost(model, tensor_shape)/1000000000)
# # 11144683520
#
# # MobileNet:
# # All: 12583738368
# # Backbone-All: 10530724864
# # Backbone-convs: 510663680 # 4.84 %
# # Backbone-ReLUs: 10020061184 # 95.1%
#
# # Decoder-All: 2053013504
# # Decoder-CoNVs: 366903296 # 17.8%
# # Decoder-ReLUs: 1686110208 # 82.1%
#
#
# # arch_utils = ArchUtilsFactory()("MobileNetV2")
# # arch_utils.set_bReLU_layers(model, layer_name_to_block_size_indices)
# # tensor_shape = [256, 3]
# # print(get_mobilenetv2_cost(model, tensor_shape) / get_mobilenetv2_cost(model_base, tensor_shape))
# # bandwidth_bytes = 12583738368
# # # get_deeplab_decoder_cost(model, [32, 3])
# # bandwidth_bytes = 11706171392
# # num_relus = 21316096
# # 12583738368 / 21316096
# # 3565682688 / 12583738368
# # # ReluCount = 21316096
# #
# #
# # deep_lab_cost = 2053013504
# # baseline_cost = 12583738368
# #
# # (3565682688 - deep_lab_cost)/(baseline_cost-deep_lab_cost)
# #
#
# # 1512669184
# # 788430336
# # 2301099520
#
# # 10530724864.0
# # 12583738368.0
#
# # Baseline:
#
# # backbone: 10530724864.0
# # Decoder: 2053013504
# # All: 12583738368.0
#
# # Mine:
#
# # backbone: 1512669184.0
# # Decoder: 788430336
# # All: 2301099520.0
#
#
#
#
# general_counter = 0
# counter = 0
# for layer_name, block_sizes in layer_name_to_block_size_indices.items():
#     for block_size in block_sizes:
#         general_counter += 1
#         if all(block_size == [0,1]):
#             counter += 1
#
#
#
#
# # MobileNet:
# All = 12583738368
# Backbone_All = 10530724864
# Backbone_convs = 510663680 # 4.84 %
# Backbone_ReLUs = 10020061184 # 95.1%
#
# Decoder_All = 2053013504
# Decoder_CoNVs = 366903296 # 17.8%
# Decoder_ReLUs = 1686110208 # 82.1%
#
# (0.12*Decoder_ReLUs + Decoder_CoNVs + 0.12*Backbone_ReLUs + Backbone_convs) / All


# m = 12
# i = 128
# f = 3
# q = m - f + 1
# o = 32
# log_p = 8
#
# conv_cost = 2 * m ** 2 * i + 2 * f ** 2 * o * i + q ** 2 * o
# relu_cost = ((6 * log_p + 19) * m ** 2) * o

# print(conv_cost / (conv_cost + relu_cost))