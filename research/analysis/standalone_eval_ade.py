import pickle
import numpy as np
import glob


def eval(results):
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    iou = total_area_intersect / total_area_union
    return iou.mean()

files = glob.glob("/home/yakir/evaluation/0.06/ade20k_lsb_0_msb_0_t_12/*")
results_secure = [tuple(np.load(x, allow_pickle=True)) for x in files]
sample_ids = [int(x.split("/")[-1].split(".")[0]) for x in files]
results_non_secure = pickle.load(open("/home/yakir/results_ade.pickle", "rb"))
results_non_secure = [results_non_secure[x] for x in sample_ids]

pre_eval_results = tuple(zip(*results_non_secure))
total_area_intersect = sum(pre_eval_results[0])
total_area_union = sum(pre_eval_results[1])
total_area_pred_label = sum(pre_eval_results[2])
total_area_label = sum(pre_eval_results[3])
iou_not_secure = total_area_intersect / total_area_union
iou_not_secure = np.nanmean(iou_not_secure.numpy())

pre_eval_results = tuple(zip(*results_secure))
total_area_intersect = sum(pre_eval_results[0])
total_area_union = sum(pre_eval_results[1])
total_area_pred_label = sum(pre_eval_results[2])
total_area_label = sum(pre_eval_results[3])
iou_secure = total_area_intersect / total_area_union
iou_secure = np.nanmean(iou_secure.numpy())
print(iou_secure, iou_not_secure, iou_secure/iou_not_secure)
print(len(files))
# 72