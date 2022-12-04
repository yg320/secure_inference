import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np

# TODO: verify max-pool
rate_coco_resnet = np.array([0.2439, 0.1972, 0.17, 0.1524, 0.1433])
band_coco_resnet = 9303332800.0 * rate_coco_resnet / 1000000000
pref_coco_resnet = np.array([99, 95, 90, 85, 82])

rate_ade_mobile = np.array([0.2139, 0.1972, 0.18, 0.134, 0.11])
band_ade_mobile = 6183918464.0 * rate_ade_mobile / 1000000000
pref_ade_mobile = np.array([98, 96, 93, 87, 81])

rate_cityscape_faster = np.array([0.2539, 0.2172, 0.17, 0.15, 0.10])
band_cityscape_faster = 10183918464.0 * rate_ade_mobile / 1000000000
pref_cityscape_faster = np.array([97, 93, 91, 88, 85])


COLOR_A = "#69b3a2"
COLOR_B = "#3399e6"
COLOR_C = "#c74c52"

plt.figure(figsize=(10,8))
plt.scatter(rate_coco_resnet, pref_coco_resnet, s=100,  color=COLOR_A)
plt.plot(rate_coco_resnet, pref_coco_resnet, lw=3, color=COLOR_A, label="DeepLabV3-ResNet51-COCO")
plt.scatter(rate_ade_mobile, pref_ade_mobile, s=100,  color=COLOR_B)
plt.plot(rate_ade_mobile, pref_ade_mobile, lw=3, color=COLOR_B, label="DeepLabV3-MobileNetV2-ADE")
plt.xlabel("Rate", fontsize=14)
plt.ylabel("Performance Relative to Baseline", fontsize=14)
plt.xticks(np.arange(0.10,0.31,0.01))
plt.yticks(np.arange(80,101,1))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim([0.10,0.30])
plt.ylim([80,100])
plt.legend()



plt.figure(figsize=(10, 8))
plt.scatter(band_coco_resnet, pref_coco_resnet, s=100, color=COLOR_A)
plt.plot(band_coco_resnet, pref_coco_resnet, lw=3, color=COLOR_A, label="DeepLabV3-ResNet51-COCO (Baseline Bandwidth = 9.3GB)")
# for i, txt in enumerate(rate_coco_resnet):
#     plt.annotate(f"R={txt}", (band_coco_resnet[i] + 0.02, pref_coco_resnet[i]), fontsize=12)

plt.scatter(band_ade_mobile, pref_ade_mobile, s=100, color=COLOR_B)
plt.plot(band_ade_mobile, pref_ade_mobile, lw=3, color=COLOR_B, label="DeepLabV3-MobileNetV2-ADE (Baseline Bandwidth = 6.2GB)")
# for i, txt in enumerate(rate_ade_mobile):
#     plt.annotate(f"R={txt}", (band_ade_mobile[i] + 0.02, pref_ade_mobile[i]), fontsize=12)

plt.scatter(band_cityscape_faster, pref_cityscape_faster, s=100, color=COLOR_C)
plt.plot(band_cityscape_faster, pref_cityscape_faster, lw=3, color=COLOR_C, label="FasterRCNN-ResNet51-Cityscape (Baseline Bandwidth = 10.1GB)")
# for i, txt in enumerate(rate_cityscape_faster):
#     plt.annotate(f"R={txt}", (band_cityscape_faster[i] + 0.02, pref_cityscape_faster[i]), fontsize=12)

plt.xlabel("Bandwidth (GB)", fontsize=18)
plt.ylabel("Performance Relative to Baseline", fontsize=18)
plt.xticks(np.arange(0.7, 2.6, 0.2))
plt.yticks(np.arange(80, 101, 2))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# plt.xlim([0.10, 0.30])
plt.ylim([80, 100])
# plt.legend()
plt.tick_params(axis='both', which='major', labelsize=18)
plt.tick_params(axis='both', which='minor', labelsize=8)
