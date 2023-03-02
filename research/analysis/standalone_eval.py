# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt

import pickle
# import mmcv
# from research.distortion.parameters.factory import param_factory
# from research.utils import build_data
import numpy as np
import glob

# CONFIG_PATH = "/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool.py"
# MODEL_PATH = "/home/yakir/iter_20000_0.06.pth"
# RELU_SPEC_FILE = '/home/yakir/assets/resnet_voc/block_spec/0.06.pickle'

# cfg = mmcv.Config.fromfile(CONFIG_PATH)
# params = param_factory(cfg)

# dataset = build_data(cfg, mode="test")


files = glob.glob("/home/yakir/voc_lsb_0_msb_0_t_12/voc_lsb_0_msb_0_t_12/*")
results_secure = [tuple(np.load(x, allow_pickle=True)) for x in files]
sample_ids = [int(x.split("/")[-1].split(".")[0]) for x in files]
results_non_secure = pickle.load(open("/home/yakir/results_voc.pickle", "rb"))
results_non_secure = [results_non_secure[x] for x in sample_ids]

# per_sample_not_secure = [dataset.evaluate([x], logger='silent', **{'metric': ['mIoU']})['mIoU'] for x in results_non_secure]
# per_sample_secure = [dataset.evaluate([x], logger='silent', **{'metric': ['mIoU']})['mIoU'] for x in results_secure]


pre_eval_results = tuple(zip(*results_non_secure))
total_area_intersect = sum(pre_eval_results[0])
total_area_union = sum(pre_eval_results[1])
total_area_pred_label = sum(pre_eval_results[2])
total_area_label = sum(pre_eval_results[3])
iou_not_secure = total_area_intersect / total_area_union
print(np.nanmean(iou_not_secure.numpy()))

pre_eval_results = tuple(zip(*results_secure))
total_area_intersect = sum(pre_eval_results[0])
total_area_union = sum(pre_eval_results[1])
total_area_pred_label = sum(pre_eval_results[2])
total_area_label = sum(pre_eval_results[3])
iou_secure = total_area_intersect / total_area_union
print(np.nanmean(iou_secure.numpy()))


#
#
# per_sample_not_secure = np.array(per_sample_not_secure)
# per_sample_secure = np.array(per_sample_secure)
# plt.scatter(per_sample_not_secure, per_sample_secure)
# plt.xlim([0,1])
# plt.ylim([0,1])
# print("not secure", dataset.evaluate(results_non_secure, logger='silent', **{'metric': ['mIoU']})['mIoU'])
# print("secure", dataset.evaluate(results_secure, logger='silent', **{'metric': ['mIoU']})['mIoU'])
