import torch
import numpy as np

from research.distortion.utils import get_model, get_data
from research.utils import build_data
from research.distortion.arch_utils.factory import arch_utils_factory
# from mmseg.ops import resize
# import torch.nn.functional as F
# from mmseg.core import intersect_and_union
import contextlib
from functools import lru_cache
import ctypes
import mmcv

@contextlib.contextmanager
def model_block_relu_transform(model, relu_spec, arch_utils):
    layer_name_to_orig_layer = {}
    for layer_name, block_sizes in relu_spec.items():
        orig_layer = arch_utils.get_layer(model, layer_name)
        layer_name_to_orig_layer[layer_name] = orig_layer

        arch_utils.set_bReLU_layers(model, {layer_name: block_sizes})

    yield model

    for layer_name_, orig_layer in layer_name_to_orig_layer.items():
        arch_utils.set_layers(model, {layer_name_: orig_layer})


IMAGES = "***"


@lru_cache(maxsize=None)
def get_num_relus(block_size, activation_dim):

    if block_size in [(0, 1), (1, 0)]:
        return 0

    if block_size == (1, 1):
        return activation_dim ** 2

    avg_pool = torch.nn.AvgPool2d(
        kernel_size=block_size,
        stride=block_size, ceil_mode=True)

    cur_input = torch.zeros(size=(1, 1, activation_dim, activation_dim))
    cur_relu_map = avg_pool(cur_input)
    num_relus = cur_relu_map.shape[2] * cur_relu_map.shape[3]

    return num_relus


def get_block_spec_num_relus(block_spec, params):
    num_relus = 0
    for layer_name, block_sizes in block_spec.items():
        activation_dim = params.LAYER_NAME_TO_DIMS[layer_name][1]
        for block_size in block_sizes:
            num_relus += get_num_relus(tuple(block_size), activation_dim)
    return num_relus


@lru_cache(maxsize=None)
def get_brelu_bandwidth(block_size, activation_dim, l=8, log_p=8, protocol="Porthos", scalar_vector_optimization=True, with_prf=True):
    assert with_prf
    if block_size in [(0, 1), (1, 0)]:
        return 0
    num_relus = get_num_relus(block_size, activation_dim)
    if protocol == "Porthos":
        dReLU_bandwidth = (6 * log_p + 19 - 5) * num_relus
    elif protocol == "SecureNN":
        dReLU_bandwidth = (8 * log_p + 24 - 5) * num_relus
    else:
        assert False
    if scalar_vector_optimization:
        assert with_prf
        mult_bandwidth = 3 * activation_dim ** 2 + 2 * num_relus
    else:
        assert with_prf
        mult_bandwidth = 5 * activation_dim ** 2
    bandwidth = l * (dReLU_bandwidth + mult_bandwidth)
    return bandwidth  # Bandwidth is in Bytes (therefore l=8 and not 64)


class DistortionUtils:
    def __init__(self, gpu_id, params, cfg, mode):

        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.params = params
        self.cfg = cfg
        self.arch_utils = arch_utils_factory(self.cfg)
        self.model = get_model(
            config=self.cfg,
            gpu_id=self.gpu_id,
            checkpoint_path=self.params.CHECKPOINT
        )

        # TODO: Replaced test with train
        # TODO: find a more elegant way to do this
        # TODO: this is copied again and again
        self.dataset = build_data(self.cfg, mode=mode)

        np.random.seed(123)
        self.shuffled_indices = np.arange(len(self.dataset))
        np.random.shuffle(self.shuffled_indices)

    def get_loss(self, out, ground_truth):
        return self.model.head.loss(out, ground_truth.to(torch.long))['loss'].cpu().numpy()
        loss_ce_list = []
        for sample_id in range(out.shape[0]):
            loss_ce_list.append(
                self.model.decode_head.losses(out[sample_id:sample_id + 1], ground_truth[sample_id:sample_id + 1])[
                    'loss_ce'].cpu().numpy())
        return loss_ce_list

    def get_samples(self, batch_index, batch_size):

        batch_indices = np.arange(batch_index * batch_size, batch_index * batch_size + batch_size)
        batch_indices = self.shuffled_indices[batch_indices]
        # TODO: just use normal training datastream.. no need to center crop then
        batch = torch.stack([self.dataset[sample_id]['img'].data for sample_id in batch_indices]).to(self.device)

        # TODO: find a more elegant way to do this
        if self.cfg.model.type == 'ImageClassifier':
            ground_truth = torch.Tensor([self.dataset[sample_id]["gt_label"] for sample_id in batch_indices]).to(self.device)
        elif self.cfg.model.type == 'EncoderDecoder':
            ground_truth = torch.stack([self.dataset[sample_id]['gt_semantic_seg'].data for sample_id in batch_indices]).to(self.device)
        elif self.cfg.model.type == 'SingleStageDetector':
            ground_truth = [self.dataset[sample_id]['gt_labels'].data.to(self.device) for sample_id in batch_indices]

        return batch, ground_truth

    @staticmethod
    def get_distortion(block_name_to_activation_baseline, block_name_to_activation_distorted):

        noises = {}
        signals = {}

        for k in block_name_to_activation_distorted.keys():
            distorted = block_name_to_activation_distorted[k]
            baseline = block_name_to_activation_baseline[k]

            dim = [1] if distorted.dim() == 2 else [1, 2, 3]

            noises[k] = ((distorted - baseline) ** 2).mean(dim=dim).cpu().numpy()
            signals[k] = (baseline ** 2).mean(dim=dim).cpu().numpy()

        return noises, signals

    def get_activations(self, block_size_spec, input_block_name, input_tensor, output_block_names, ground_truth):
        with torch.no_grad():
            with model_block_relu_transform(self.model, block_size_spec, self.arch_utils) as model:
                torch.cuda.empty_cache()
                input_block_index = np.argwhere(np.array(self.params.BLOCK_NAMES) == input_block_name)[0, 0]

                output_block_indices = [np.argwhere(np.array(self.params.BLOCK_NAMES) == output_block_name)[0, 0] for
                                        output_block_name in output_block_names]

                block_name_to_activation = dict()
                activation = input_tensor

                for block_index in range(input_block_index, max(output_block_indices) + 1):
                    block_name = self.params.BLOCK_NAMES[block_index]
                    activation = self.arch_utils.run_model_block(model, activation, self.params.BLOCK_NAMES[block_index])

                    if block_name in output_block_names:
                        block_name_to_activation[block_name] = activation

                if True: #output_block_names[-1] == self.params.BLOCK_NAMES[-2]:  # TODO: why do we have None in the end of self.BLOCK_NAMES?
                    losses = self.get_loss(activation, ground_truth)
                else:
                    losses = np.nan * np.ones(shape=(input_tensor.shape[0],))

                return block_name_to_activation, losses

    @lru_cache(maxsize=1)
    def get_batch_data(self, batch_index, batch_size, block_size_spec_id):
        block_size_spec = ctypes.cast(block_size_spec_id, ctypes.py_object).value
        input_images, ground_truth = self.get_samples(batch_index, batch_size=batch_size)

        block_name_to_activation, losses = \
            self.get_activations(block_size_spec,
                                 input_block_name=self.params.BLOCK_NAMES[0],
                                 input_tensor=input_images,
                                 output_block_names=self.params.BLOCK_NAMES[:-1],
                                 ground_truth=ground_truth)
        block_name_to_activation["input_images"] = input_images
        return block_name_to_activation, ground_truth, losses

    def get_batch_distortion(self, baseline_block_size_spec, block_size_spec, batch_index, batch_size, input_block_name,
                             output_block_name):

        block_name_to_activation_baseline, ground_truth, losses_baseline = \
            self.get_batch_data(batch_index, batch_size, id(baseline_block_size_spec))

        input_tensor = block_name_to_activation_baseline[self.params.BLOCK_INPUT_DICT[input_block_name]]

        block_name_to_activation_distorted, losses_distorted = \
            self.get_activations(block_size_spec,
                                 input_block_name=input_block_name,
                                 input_tensor=input_tensor,
                                 output_block_names=[output_block_name],
                                 ground_truth=ground_truth)

        noises, signals = self.get_distortion(
            block_name_to_activation_baseline=block_name_to_activation_baseline,
            block_name_to_activation_distorted=block_name_to_activation_distorted)

        assets = {
            "Baseline Loss": float(losses_baseline),
            "Distorted Loss": float(losses_distorted),
            "Noise": noises[output_block_name].mean(),
            "Signal": signals[output_block_name].mean(),
        }
        return assets

