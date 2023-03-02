import pickle
import numpy as np
import glob


def evaluate(results):
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    iou = total_area_intersect / total_area_union
    return np.nanmean(iou)

files = glob.glob("/home/yakir/evaluation/0.06/ade20k_lsb_0_msb_0_t_12/*")
results_secure = [tuple(np.load(x, allow_pickle=True)) for x in files]
sample_ids = [int(x.split("/")[-1].split(".")[0]) for x in files]
results_non_secure = pickle.load(open("/home/yakir/results_ade.pickle", "rb"))
results_non_secure = [results_non_secure[x] for x in sample_ids]

x = -1
for i in range(len(files)):
    iou_not_secure = evaluate(results_non_secure[:i] + results_non_secure[i+1:])
    iou_secure = evaluate(results_secure[:i] + results_secure[i+1:])
    x = max(x, iou_secure/iou_not_secure)
    # print(iou_secure, iou_not_secure, iou_secure/iou_not_secure)
print(x)
print(len(files))
# 72