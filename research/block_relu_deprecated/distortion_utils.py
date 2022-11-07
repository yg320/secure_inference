import torch
import numpy as np

from research.block_relu.utils import get_model, get_data, center_crop, ArchUtilsFactory

from mmseg.ops import resize
import torch.nn.functional as F
from mmseg.core import intersect_and_union
import contextlib
from research.block_relu.params import MobileNetV2Params
from functools import lru_cache


@contextlib.contextmanager
def model_block_relu_transform(model, relu_spec, arch_utils, params):
    layer_name_to_orig_layer = {}
    for layer_name, block_sizes in relu_spec.items():
        orig_layer = arch_utils.get_layer(model, layer_name)
        layer_name_to_orig_layer[layer_name] = orig_layer

        arch_utils.set_bReLU_layers(model, {layer_name: block_sizes})

    yield model

    for layer_name_, orig_layer in layer_name_to_orig_layer.items():
        arch_utils.set_layers(model, {layer_name_: orig_layer})


IMAGES = "***"


class DistortionUtils:
    def __init__(self, gpu_id, params):

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.params = params
        self.deformation_base_path = "/home/yakir/tmp_d"

        self.arch_utils = ArchUtilsFactory()(params.BACKBONE)
        self.model = get_model(
            config=params.CONFIG,
            gpu_id=self.gpu_id,
            checkpoint_path=params.CHECKPOINT
        )

        self.ds_name = params.DATASET
        self.dataset = get_data(self.ds_name)
        np.random.seed(123)
        self.shuffled_indices = np.arange(len(self.dataset))
        np.random.shuffle(self.shuffled_indices)

    # def get_mIoU(self, out, ground_truth):
    #     seg_logit = resize(
    #         input=out,
    #         size=(ground_truth.shape[2], ground_truth.shape[3]),
    #         mode='bilinear',
    #         align_corners=self.model.decode_head.align_corners)
    #     output = F.softmax(seg_logit, dim=1)
    #
    #     seg_pred = output.argmax(dim=1).cpu().numpy()
    #     gt = ground_truth[:, 0].cpu().numpy()
    #
    #     mIoUs = []
    #     for sample in range(seg_pred.shape[0]):
    #         results = [intersect_and_union(
    #             seg_pred[sample:sample + 1],
    #             gt[sample:sample + 1],
    #             len(self.dataset.CLASSES),
    #             self.dataset.ignore_index,
    #             label_map=dict(),
    #             reduce_zero_label=False)]
    #         assert self.ds_name == "ade_20k"
    #
    #         mIoUs.append(self.dataset.evaluate(results, logger='silent', **{'metric': ['mIoU']})['mIoU'])
    #
    #     return mIoUs

    def get_loss(self, out, ground_truth):
        # loss_ce = self.model.decode_head.losses(out, ground_truth)['loss_ce'].cpu().numpy()
        loss_ce_list = []
        for sample_id in range(out.shape[0]):
            loss_ce_list.append(
                self.model.decode_head.losses(out[sample_id:sample_id + 1], ground_truth[sample_id:sample_id + 1])[
                    'loss_ce'].cpu().numpy())
        return loss_ce_list

    def get_samples(self, batch_indices, im_size=512):
        batch = torch.stack(
            [center_crop(self.dataset[sample_id]['img'].data, im_size) for sample_id in batch_indices]).to(self.device)
        ground_truth = torch.stack(
            [center_crop(self.dataset[sample_id]['gt_semantic_seg'].data, im_size) for sample_id in batch_indices]).to(
            self.device)
        return batch, ground_truth

    def get_activations(self, input_block_name, input_tensor, output_block_names, model, ground_truth=None):
        torch.cuda.empty_cache()
        input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]

        output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for
                                output_block_name in output_block_names]

        resnet_block_name_to_activation = dict()
        activation = input_tensor

        for block_index in range(input_block_index, max(output_block_indices) + 1):
            block_name = self.params.BLOCK_NAMES[block_index]
            activation = self.arch_utils.run_model_block(model, activation, self.params.BLOCK_NAMES[block_index])

            if block_name in output_block_names:
                resnet_block_name_to_activation[block_name] = activation

        if output_block_names[-1] == "decode":
            mIoUs = [] #self.get_mIoU(activation, ground_truth)
            losses = self.get_loss(activation, ground_truth)
        else:
            mIoUs = []
            losses = []

        return resnet_block_name_to_activation, losses, mIoUs

    def get_inputs(self, block_size_spec, batch, resnet_block_name_to_activation_baseline):
        # assert len(block_size_spec) in [1, 40]
        earliest_layer_name = self.params.LAYER_NAMES[(np.argwhere(
            (np.array(list(block_size_spec.keys()))[:, np.newaxis] == np.array(self.params.LAYER_NAMES)[np.newaxis]))[:,
                                                       1]).min()]
        layer_block = self.params.LAYER_NAME_TO_BLOCK_NAME[earliest_layer_name]
        layer_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == layer_block)[0, 0]
        input_block_name = self.params.BLOCK_NAMES[layer_block_index]
        output_block_names = self.params.BLOCK_NAMES[layer_block_index:-1]
        if layer_block_index == 0:
            input_tensor = batch
        else:
            prev_block_name = self.params.BLOCK_NAMES[layer_block_index - 1]
            input_tensor = resnet_block_name_to_activation_baseline[prev_block_name]

        return input_tensor, input_block_name, output_block_names

    def get_distortion(self, resnet_block_name_to_activation_baseline, resnet_block_name_to_activation_distorted):

        noises = {}
        signals = {}

        for k in resnet_block_name_to_activation_distorted.keys():
            distorted = resnet_block_name_to_activation_distorted[k]
            baseline = resnet_block_name_to_activation_baseline[k]

            noises[k] = ((distorted - baseline) ** 2).mean(dim=[1, 2, 3]).cpu().numpy()
            signals[k] = (baseline ** 2).mean(dim=[1, 2, 3]).cpu().numpy()

        return noises, signals

    def get_batch_distortion(self, block_size_spec, batch_index, batch_size=8, im_size=512, output_block_names=None):

        with torch.no_grad():
            torch.cuda.empty_cache()

            batch, ground_truth, resnet_block_name_to_activation_baseline, losses_baseline, mIoUs_baseline = \
                self.get_samples_and_activations(batch=batch_index, batch_size=batch_size, im_size=im_size)

            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils, self.params) as noisy_model:
                input_tensor, input_block_name, suggested_output_block_names = self.get_inputs(block_size_spec, batch,
                                                                                               resnet_block_name_to_activation_baseline)

                if output_block_names is None:
                    output_block_names = suggested_output_block_names
                resnet_block_name_to_activation_distorted, losses_distorted, mIoUs_distorted = \
                    self.get_activations(input_block_name=input_block_name,
                                         input_tensor=input_tensor,
                                         output_block_names=output_block_names,
                                         model=noisy_model,
                                         ground_truth=ground_truth)

            noises_distorted, signals_distorted = self.get_distortion(
                resnet_block_name_to_activation_baseline=resnet_block_name_to_activation_baseline,
                resnet_block_name_to_activation_distorted=resnet_block_name_to_activation_distorted)

        assets = {
            "losses_baseline": losses_baseline,
            "losses_distorted": losses_distorted,
            "mIoUs_baseline": mIoUs_baseline,
            "mIoUs_distorted": mIoUs_distorted,
            "noises_distorted": noises_distorted,
            "signals_distorted": signals_distorted,
        }
        return assets

    @lru_cache(maxsize=1)
    def get_samples_and_activations(self, batch, batch_size, im_size):
        batch_indices = np.arange(batch * batch_size, batch * batch_size + batch_size)
        batch, ground_truth = self.get_samples(self.shuffled_indices[batch_indices], im_size=im_size)
        resnet_block_name_to_activation_baseline, cur_losses, cur_mIoUs = \
            self.get_activations(input_block_name=self.params.BLOCK_NAMES[0],
                                 input_tensor=batch,
                                 output_block_names=self.params.BLOCK_NAMES[:-1],
                                 model=self.model,
                                 ground_truth=ground_truth)

        return batch, ground_truth, resnet_block_name_to_activation_baseline, cur_losses, cur_mIoUs
