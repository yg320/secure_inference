# DenseCRF
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from utils import crf
import numpy as np
from cv2 import imread, imwrite
from tqdm import tqdm
import os


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)


bi_xy_std = np.random.uniform(50, 100)
w1 = np.random.uniform(5, 10)
std_beta = np.random.uniform(3, 10)

basedir = os.path.join(f"/home/yakir/tmp/{bi_xy_std:.5}_{w1:.5}_{std_beta:.5}")
os.makedirs(basedir)
post_processor = crf.DenseCRF(
    iter_max=10,
    pos_xy_std=3,
    pos_w=3,

    bi_xy_std=bi_xy_std,  # 50, 60, 70, 80, 90, 100
    bi_rgb_std=std_beta,  # 3, 4, 5, 6, 7, 8, 9, 10
    bi_w=w1  # 5, 6, 7, 8 , 9, 10
)

for image_index in tqdm(range(1, 501)):
    str_image_index = str(image_index).zfill(8)
    IM_FILE = f'/home/yakir/Data/ade/ADEChallengeData2016/images/validation/ADE_val_{str_image_index}.jpg'
    LOGIT_FILE = f'/home/yakir/Data2/logits_192/ADE_val_{str_image_index}.npy'
    ANNOT = f'/home/yakir/Data/ade/ADEChallengeData2016/annotations/validation/ADE_val_{str_image_index}.png'

    logits = np.load(LOGIT_FILE)[0]
    img = imread(IM_FILE)[..., ::-1]
    annot = imread(ANNOT)

    classes = np.unique(logits.argmax(axis=0))
    probs_orig = softmax(logits[classes])

    prob_crf = post_processor(img, probs_orig)

    orig = np.argmax(probs_orig, axis=0)
    refined = np.argmax(prob_crf, axis=0)

    orig = classes[orig.flatten()].reshape(orig.shape)
    refined = classes[refined.flatten()].reshape(refined.shape)

    np.save(file=os.path.join(basedir, f"orig_{str_image_index}.npy"), arr=orig)
    np.save(file=os.path.join(basedir, f"refined_{str_image_index}.npy"), arr=refined)
