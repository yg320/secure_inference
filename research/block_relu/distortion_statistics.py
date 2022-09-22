import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import torch
import numpy as np
import pickle

from research.block_relu.utils import get_model, get_data, center_crop, ArchUtilsFactory
from research.block_relu.params import ParamsFactory

from research.pipeline.backbones.secure_resnet import MyResNet # TODO: find better way to init
from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union


class DistortionStatistics:
    def __init__(self, gpu_id, param_json_file, batch_size=8):

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"

        self.params = ParamsFactory()(param_json_file)
        self.deformation_base_path = os.path.join("/home/yakir/Data2/assets_v3/distortion_statistics_exploration", self.params.DATASET, self.params.BACKBONE)

        self.arch_utils = ArchUtilsFactory()(self.params.BACKBONE)

        self.batch_size = batch_size
        self.im_size = 512

        self.model = get_model(
            config=self.params.CONFIG,
            gpu_id=self.gpu_id,
            checkpoint_path=self.params.CHECKPOINT
        )
        self.dataset = get_data(self.params.DATASET)

    def get_activations(self, batch_index):

        num_blocks = len(self.params.BLOCK_NAMES) - 1

        with torch.no_grad():
            batch_indices = range(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
            batch = torch.stack(
                [center_crop(self.dataset[sample_id]['img'].data, self.im_size) for sample_id in batch_indices]).to(
                self.device)
            ground_truth = torch.stack(
                [center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, self.im_size) for sample_id in
                 batch_indices]).to(self.device)
            activations = [batch]
            for block_index in range(num_blocks):
                activations.append(self.arch_utils.run_model_block(self.model, activations[block_index], self.params.BLOCK_NAMES[block_index]))
            loss_ce = self.model.decode_head.losses(activations[-1], ground_truth)['loss_ce'].cpu()
            resnet_block_name_to_activation = dict(zip(self.params.BLOCK_NAMES, activations))

            return resnet_block_name_to_activation, ground_truth, loss_ce

    def _get_deformation_v2(self, resnet_block_name_to_activation, ground_truth, block_size_spec, input_block_name, output_block_names):

        input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]
        output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for output_block_name in output_block_names]

        input_tensor = resnet_block_name_to_activation[input_block_name]
        next_tensors = [resnet_block_name_to_activation[output_block_name] for output_block_name in output_block_names]

        layer_name_to_orig_layer = {}
        for layer_name, block_size_indices in block_size_spec.items():
            orig_layer = self.arch_utils.get_layer(self.model, layer_name)
            layer_name_to_orig_layer[layer_name] = orig_layer

            self.arch_utils.set_bReLU_layers(self.model, {layer_name: (block_size_indices,
                                                                       self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])})

        cur_out = input_tensor
        outs = []

        for block_index in range(input_block_index, max(output_block_indices)):
            cur_out = self.arch_utils.run_model_block(self.model, cur_out, self.params.BLOCK_NAMES[block_index])
            outs.append(cur_out)

        if output_block_names[-1] is None:
            loss_deform = self.model.decode_head.losses(cur_out, ground_truth)['loss_ce']
        else:
            loss_deform = None

        for layer_name_, orig_layer in layer_name_to_orig_layer.items():
            self.arch_utils.set_layers(self.model, {layer_name_: orig_layer})


        seg_logit = resize(
            input=outs[-1],
            size=(512, 512),
            mode='bilinear',
            align_corners=self.model.decode_head.align_corners)
        output = F.softmax(seg_logit, dim=1)

        seg_pred = output.argmax(dim=1)

        results = [intersect_and_union(
            seg_pred.cpu().numpy(),
            ground_truth[0].cpu().numpy(),
            len(self.dataset.CLASSES),
            self.dataset.ignore_index,
            label_map=dict(),
            reduce_zero_label=self.dataset.reduce_zero_label)]

        mIoU = self.dataset.evaluate(results, logger = 'silent', **{'metric': ['mIoU']})['mIoU']

        noises = []
        signals = []

        for out, next_tensor in zip(outs, next_tensors):
            noise = float(((out - next_tensor) ** 2).mean())
            signal = float((next_tensor ** 2).mean())
            noises.append(noise)
            signals.append(signal)

        return noises, signals, loss_deform.cpu().numpy(), mIoU.cpu().numpy()

    def get_random_spec(self):
        channel_ord_to_layer_name = np.hstack([self.params.LAYER_NAME_TO_CHANNELS[layer_name] * [layer_name] for layer_name in self.params.LAYER_NAMES])
        channel_ord_to_channel_index = np.hstack([np.arange(self.params.LAYER_NAME_TO_CHANNELS[layer_name]) for layer_name in self.params.LAYER_NAMES])
        num_channels = len(channel_ord_to_layer_name)

        layers_to_noise = ["stem_2",
                           "stem_5",
                           "stem_8",
                           "layer1_0_1",
                           "layer1_0_2",
                           "layer1_0_3",
                           "layer1_1_1",
                           "layer1_1_2",
                           "layer1_1_3",
                           "layer1_2_1",
                           "layer1_2_2",
                           "layer1_2_3",
                           "layer2_0_1",
                           "layer2_0_2",
                           "layer2_0_3",
                           "layer2_1_1",
                           "layer2_1_2",
                           "layer2_1_3",
                           "layer2_2_1",
                           "layer2_2_2",
                           "layer2_2_3",
                           "layer2_3_1",
                           "layer2_3_2",
                           "layer2_3_3", ]

        num_channel_to_add_noise_to = num_channels #// 2  # np.random.randint(low=0, high=num_channels)
        channel_sample = np.random.choice(num_channels, size=num_channel_to_add_noise_to, replace=False)
        layer_and_channels = [(channel_ord_to_layer_name[x], channel_ord_to_channel_index[x]) for x in channel_sample]
        # block_size_pseudo_indices = np.random.choice(5, size=num_channel_to_add_noise_to, replace=True)
        block_size_indices = [np.random.randint(1, len(self.params.LAYER_NAME_TO_BLOCK_SIZES[layer_name])) for
                              layer_name, _ in layer_and_channels]
        block_size_spec = {
            layer_name: np.zeros(shape=(self.params.LAYER_NAME_TO_CHANNELS[layer_name],), dtype=np.int32) for
            layer_name in self.params.LAYER_NAMES if layer_name in layers_to_noise}

        for index, (layer_name, channel) in zip(block_size_indices, layer_and_channels):
            if layer_name in layers_to_noise:
                block_size_spec[layer_name][channel] = index

        return block_size_spec

    def explore_noise_correlation(self):
        self.batch_size = 1
        os.makedirs(self.deformation_base_path, exist_ok=True)
        with torch.no_grad():

            torch.cuda.empty_cache()

            for batch_index in tqdm(range(len(self.dataset))):
                try:
                    resnet_block_name_to_activation, ground_truth, loss_ce = self.get_activations(batch_index)
                except ValueError:
                    continue

                block_size_spec = self.get_random_spec()

                noises, signals, loss_deform, mIoU = self._get_deformation_v2(
                    resnet_block_name_to_activation=resnet_block_name_to_activation,
                    ground_truth=ground_truth,
                    block_size_spec=block_size_spec,
                    input_block_name="stem",
                    output_block_names=[None])




if __name__ == '__main__':
    dh = DistortionStatistics(gpu_id=0,
                              param_json_file="/home/yakir/PycharmProjects/secure_inference/research/block_relu/distortion_handler_configs/resnet_COCO_164K_8_hierarchies.json")

    dh.explore_noise_correlation()