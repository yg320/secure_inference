## What does this package offer?:

## How To Run Secure KSNet:
###  [Segmentation, MobileNetV2, COCO](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py)
- **First, we extract disotrtion for each channel and each block size by running:** 
    - python research/extract_block_sizes.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py --checkpoint {PATH_TO_MMLAB_MODELS}/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --output_path {WORK_DIR}/segmentation --num_samples NUM_SAMPLES --num_gpus NUM_GPUS 
    - we used NUM_SAMPLES=48 over NUM_GPUS=4
- **Now we are ready to get the knapsack optimal patch-sizes by running:**
  - export PYTHONPATH=. ; python research/distortion/knapsack/knapsack_patch_size_extractor.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py --block_size_spec_file_name benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle --channel_distortion_path /storage/yakir/secure_inference/benchmark/segmentation/mobilenet_ade/distortion/distortion_collected/ --ratio 0.06
- **Finally, we can train the network**
  - export PYTHONPATH=. ; bash ./research/mmlab_tools/segmentation/dist_train.sh research/configs/segmentation/mobilenet_v2/deeplabev2_mobilenet_ade20k_finetune.py 4 --load-from mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --work-dir benchmark/segmentation/mobilenet_ade/experiments/0.06 --relu-spec-file benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle
 
### Classification, ResNet50, ImageNet
- **Here, we first need to replace the MaxPool layer with an AvgPool layer and finetune by running:**
    - ./research/mmlab_tools/classification/dist_train_cls.sh research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py 4 --load-from mmlab_models/classification/resnet50_8xb32_in1k_20210831-ea4938fc.pth --work-dir benchmark/classification/resnet50_coco/avg_pool
- **Next, we extract disotrtion for each channel and each block size by running:** 
    - python research/extract_block_sizes.py --config research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py --checkpoint benchmark/classification/resnet50_coco/avg_pool/epoch_15.pth --output_path {WORK_DIR}/classification --num_samples NUM_SAMPLES --num_gpus NUM_GPUS 
    - we used NUM_SAMPLES=512 over NUM_GPUS=4
- **Now we are ready to get the knapsack optimal patch-sizes by running:**
    - export PYTHONPATH=. ; python research/distortion/knapsack/knapsack_patch_size_extractor.py --config research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py --block_size_spec_file_name benchmark/classification/resnet50_coco/distortion/block_speck/0.06.pickle --channel_distortion_path /storage/yakir/secure_inference/benchmark/classification/resnet50_coco/distortion/distortion_collected --ratio 0.06
- **Finally, we can train the network**
    - export PYTHONPATH=. ; bash ./research/mmlab_tools/classification/dist_train_cls.sh research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_finetune.py 4 --load-from benchmark/classification/resnet50_coco/avg_pool/epoch_15.pth --work-dir benchmark/classification/resnet50_coco/experiments/0.06 --relu-spec-file benchmark/classification/resnet50_coco/distortion/block_speck/0.06.pickle

### Classification, ResNet18, COCO100

## Extending Secure Inference
To extend secure inference to your own architecture

- **Add distortion parameters file to distortion/parameters**
- **Add the proper line to distortion/parameters/factory.py**
- **Add the proper file to distortion/arch_utils**
- **Add the proper line to distortion/arch_utils/factor.py**
- **distortion extraction line in data**

## Next Steps:
- **Knowledge distillation (similar to DeepReDuce) with out bReLU layer and Knapsack algorithm**
- **Knapsack as a starting point for some iterative algorithm (such as simulated annealing)**
- **For larger networks, where it takes too much time to extract distortion, we can measure distortion in some middle layer and normalize by the appropriate signal to get a SNR measure, this is already supported, and a PoC has been made**
- **Iterative Knapsack, where we work on a bunch of layers each time**


[//]: # (<div align="center">)

[//]: # (  <img src="resources/mmseg-logo.png" width="600"/>)

[//]: # (  <div>&nbsp;</div>)

[//]: # (  <div align="center">)

[//]: # (    <b><font size="5">OpenMMLab website</font></b>)

[//]: # (    <sup>)

[//]: # (      <a href="https://openmmlab.com">)

[//]: # (        <i><font size="4">HOT</font></i>)

[//]: # (      </a>)

[//]: # (    </sup>)

[//]: # (    &nbsp;&nbsp;&nbsp;&nbsp;)

[//]: # (    <b><font size="5">OpenMMLab platform</font></b>)

[//]: # (    <sup>)

[//]: # (      <a href="https://platform.openmmlab.com">)

[//]: # (        <i><font size="4">TRY IT OUT</font></i>)

[//]: # (      </a>)

[//]: # (    </sup>)

[//]: # (  </div>)

[//]: # (  <div>&nbsp;</div>)

[//]: # ()
[//]: # (<br />)
[//]: # ()
[//]: # ([![PyPI - Python Version]&#40;https://img.shields.io/pypi/pyversions/mmsegmentation&#41;]&#40;https://pypi.org/project/mmsegmentation/&#41;)

[//]: # ([![PyPI]&#40;https://img.shields.io/pypi/v/mmsegmentation&#41;]&#40;https://pypi.org/project/mmsegmentation&#41;)

[//]: # ([![docs]&#40;https://img.shields.io/badge/docs-latest-blue&#41;]&#40;https://mmsegmentation.readthedocs.io/en/latest/&#41;)

[//]: # ([![badge]&#40;https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/actions&#41;)

[//]: # ([![codecov]&#40;https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/open-mmlab/mmsegmentation&#41;)

[//]: # ([![license]&#40;https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE&#41;)

[//]: # ([![issue resolution]&#40;https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/issues&#41;)

[//]: # ([![open issues]&#40;https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg&#41;]&#40;https://github.com/open-mmlab/mmsegmentation/issues&#41;)

[//]: # ()
[//]: # ([📘Documentation]&#40;https://mmsegmentation.readthedocs.io/en/latest/&#41; |)

[//]: # ([🛠️Installation]&#40;https://mmsegmentation.readthedocs.io/en/latest/get_started.html&#41; |)

[//]: # ([👀Model Zoo]&#40;https://mmsegmentation.readthedocs.io/en/latest/model_zoo.html&#41; |)

[//]: # ([🆕Update News]&#40;https://mmsegmentation.readthedocs.io/en/latest/changelog.html&#41; |)

[//]: # ([🤔Reporting Issues]&#40;https://github.com/open-mmlab/mmsegmentation/issues/new/choose&#41;)

[//]: # ()
[//]: # (</div>)

[//]: # ()
[//]: # (<div align="center">)

[//]: # ()
[//]: # (English | [简体中文]&#40;README_zh-CN.md&#41;)

[//]: # ()
[//]: # (</div>)


[//]: # ()
[//]: # (MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.)

[//]: # (It is a part of the [OpenMMLab]&#40;https://openmmlab.com/&#41; project.)

[//]: # ()
[//]: # (The master branch works with **PyTorch 1.5+**.)

[//]: # ()
[//]: # (![demo image]&#40;resources/seg_demo.gif&#41;)

[//]: # ()
[//]: # (<details open>)

[//]: # (<summary>Major features</summary>)

[//]: # ()
[//]: # (- **Unified Benchmark**)

[//]: # ()
[//]: # (  We provide a unified benchmark toolbox for various semantic segmentation methods.)

[//]: # ()
[//]: # (- **Modular Design**)

[//]: # ()
[//]: # (  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.)

[//]: # ()
[//]: # (- **Support of multiple methods out of box**)

[//]: # ()
[//]: # (  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.)

[//]: # ()
[//]: # (- **High efficiency**)

[//]: # ()
[//]: # (  The training speed is faster than or comparable to other codebases.)

[//]: # ()
[//]: # (</details>)

[//]: # ()
[//]: # (## What's New)

[//]: # ()
[//]: # (### 💎 Stable version)

[//]: # ()
[//]: # (v0.30.0 was released on 01/11/2023:)

[//]: # ()
[//]: # (- Add 'Projects/' folder, and the first example project)

[//]: # (- Support Delving into High-Quality Synthetic Face Occlusion Segmentation Datasets)

[//]: # ()
[//]: # (Please refer to [changelog.md]&#40;docs/en/changelog.md&#41; for details and release history.)

[//]: # ()
[//]: # (### 🌟 Preview of 1.x version)

[//]: # ()
[//]: # (A brand new version of **MMSegmentation v1.0.0rc3** was released in 12/31/2022:)

[//]: # ()
[//]: # (- Unifies interfaces of all components based on [MMEngine]&#40;https://github.com/open-mmlab/mmengine&#41;.)

[//]: # (- Faster training and testing speed with complete support of mixed precision training.)

[//]: # (- Refactored and more flexible [architecture]&#40;https://mmsegmentation.readthedocs.io/en/1.x/overview.html&#41;.)

[//]: # ()
[//]: # (Find more new features in [1.x branch]&#40;https://github.com/open-mmlab/mmsegmentation/tree/1.x&#41;. Issues and PRs are welcome!)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (Please refer to [get_started.md]&#40;docs/en/get_started.md#installation&#41; for installation and [dataset_prepare.md]&#40;docs/en/dataset_prepare.md#prepare-datasets&#41; for dataset preparation.)

[//]: # ()
[//]: # (## Get Started)

[//]: # ()
[//]: # (Please see [train.md]&#40;docs/en/train.md&#41; and [inference.md]&#40;docs/en/inference.md&#41; for the basic usage of MMSegmentation.)

[//]: # (There are also tutorials for:)

[//]: # ()
[//]: # (- [customizing dataset]&#40;docs/en/tutorials/customize_datasets.md&#41;)

[//]: # (- [designing data pipeline]&#40;docs/en/tutorials/data_pipeline.md&#41;)

[//]: # (- [customizing modules]&#40;docs/en/tutorials/customize_models.md&#41;)

[//]: # (- [customizing runtime]&#40;docs/en/tutorials/customize_runtime.md&#41;)

[//]: # (- [training tricks]&#40;docs/en/tutorials/training_tricks.md&#41;)

[//]: # (- [useful tools]&#40;docs/en/useful_tools.md&#41;)

[//]: # ()
[//]: # (A Colab tutorial is also provided. You may preview the notebook [here]&#40;demo/MMSegmentation_Tutorial.ipynb&#41; or directly [run]&#40;https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb&#41; on Colab.)

[//]: # ()
[//]: # (## Benchmark and model zoo)

[//]: # ()
[//]: # (Results and models are available in the [model zoo]&#40;docs/en/model_zoo.md&#41;.)

[//]: # ()
[//]: # (Supported backbones:)

[//]: # ()
[//]: # (- [x] ResNet &#40;CVPR'2016&#41;)

[//]: # (- [x] ResNeXt &#40;CVPR'2017&#41;)

[//]: # (- [x] [HRNet &#40;CVPR'2019&#41;]&#40;configs/hrnet&#41;)

[//]: # (- [x] [ResNeSt &#40;ArXiv'2020&#41;]&#40;configs/resnest&#41;)

[//]: # (- [x] [MobileNetV2 &#40;CVPR'2018&#41;]&#40;configs/mobilenet_v2&#41;)

[//]: # (- [x] [MobileNetV3 &#40;ICCV'2019&#41;]&#40;configs/mobilenet_v3&#41;)

[//]: # (- [x] [Vision Transformer &#40;ICLR'2021&#41;]&#40;configs/vit&#41;)

[//]: # (- [x] [Swin Transformer &#40;ICCV'2021&#41;]&#40;configs/swin&#41;)

[//]: # (- [x] [Twins &#40;NeurIPS'2021&#41;]&#40;configs/twins&#41;)

[//]: # (- [x] [BEiT &#40;ICLR'2022&#41;]&#40;configs/beit&#41;)

[//]: # (- [x] [ConvNeXt &#40;CVPR'2022&#41;]&#40;configs/convnext&#41;)

[//]: # (- [x] [MAE &#40;CVPR'2022&#41;]&#40;configs/mae&#41;)

[//]: # (- [x] [PoolFormer &#40;CVPR'2022&#41;]&#40;configs/poolformer&#41;)

[//]: # ()
[//]: # (Supported methods:)

[//]: # ()
[//]: # (- [x] [FCN &#40;CVPR'2015/TPAMI'2017&#41;]&#40;configs/fcn&#41;)

[//]: # (- [x] [ERFNet &#40;T-ITS'2017&#41;]&#40;configs/erfnet&#41;)

[//]: # (- [x] [UNet &#40;MICCAI'2016/Nat. Methods'2019&#41;]&#40;configs/unet&#41;)

[//]: # (- [x] [PSPNet &#40;CVPR'2017&#41;]&#40;configs/pspnet&#41;)

[//]: # (- [x] [DeepLabV3 &#40;ArXiv'2017&#41;]&#40;configs/deeplabv3&#41;)

[//]: # (- [x] [BiSeNetV1 &#40;ECCV'2018&#41;]&#40;configs/bisenetv1&#41;)

[//]: # (- [x] [PSANet &#40;ECCV'2018&#41;]&#40;configs/psanet&#41;)

[//]: # (- [x] [DeepLabV3+ &#40;CVPR'2018&#41;]&#40;configs/deeplabv3plus&#41;)

[//]: # (- [x] [UPerNet &#40;ECCV'2018&#41;]&#40;configs/upernet&#41;)

[//]: # (- [x] [ICNet &#40;ECCV'2018&#41;]&#40;configs/icnet&#41;)

[//]: # (- [x] [NonLocal Net &#40;CVPR'2018&#41;]&#40;configs/nonlocal_net&#41;)

[//]: # (- [x] [EncNet &#40;CVPR'2018&#41;]&#40;configs/encnet&#41;)

[//]: # (- [x] [Semantic FPN &#40;CVPR'2019&#41;]&#40;configs/sem_fpn&#41;)

[//]: # (- [x] [DANet &#40;CVPR'2019&#41;]&#40;configs/danet&#41;)

[//]: # (- [x] [APCNet &#40;CVPR'2019&#41;]&#40;configs/apcnet&#41;)

[//]: # (- [x] [EMANet &#40;ICCV'2019&#41;]&#40;configs/emanet&#41;)

[//]: # (- [x] [CCNet &#40;ICCV'2019&#41;]&#40;configs/ccnet&#41;)

[//]: # (- [x] [DMNet &#40;ICCV'2019&#41;]&#40;configs/dmnet&#41;)

[//]: # (- [x] [ANN &#40;ICCV'2019&#41;]&#40;configs/ann&#41;)

[//]: # (- [x] [GCNet &#40;ICCVW'2019/TPAMI'2020&#41;]&#40;configs/gcnet&#41;)

[//]: # (- [x] [FastFCN &#40;ArXiv'2019&#41;]&#40;configs/fastfcn&#41;)

[//]: # (- [x] [Fast-SCNN &#40;ArXiv'2019&#41;]&#40;configs/fastscnn&#41;)

[//]: # (- [x] [ISANet &#40;ArXiv'2019/IJCV'2021&#41;]&#40;configs/isanet&#41;)

[//]: # (- [x] [OCRNet &#40;ECCV'2020&#41;]&#40;configs/ocrnet&#41;)

[//]: # (- [x] [DNLNet &#40;ECCV'2020&#41;]&#40;configs/dnlnet&#41;)

[//]: # (- [x] [PointRend &#40;CVPR'2020&#41;]&#40;configs/point_rend&#41;)

[//]: # (- [x] [CGNet &#40;TIP'2020&#41;]&#40;configs/cgnet&#41;)

[//]: # (- [x] [BiSeNetV2 &#40;IJCV'2021&#41;]&#40;configs/bisenetv2&#41;)

[//]: # (- [x] [STDC &#40;CVPR'2021&#41;]&#40;configs/stdc&#41;)

[//]: # (- [x] [SETR &#40;CVPR'2021&#41;]&#40;configs/setr&#41;)

[//]: # (- [x] [DPT &#40;ArXiv'2021&#41;]&#40;configs/dpt&#41;)

[//]: # (- [x] [Segmenter &#40;ICCV'2021&#41;]&#40;configs/segmenter&#41;)

[//]: # (- [x] [SegFormer &#40;NeurIPS'2021&#41;]&#40;configs/segformer&#41;)

[//]: # (- [x] [K-Net &#40;NeurIPS'2021&#41;]&#40;configs/knet&#41;)

[//]: # ()
[//]: # (Supported datasets:)

[//]: # ()
[//]: # (- [x] [Cityscapes]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes&#41;)

[//]: # (- [x] [PASCAL VOC]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc&#41;)

[//]: # (- [x] [ADE20K]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k&#41;)

[//]: # (- [x] [Pascal Context]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context&#41;)

[//]: # (- [x] [COCO-Stuff 10k]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k&#41;)

[//]: # (- [x] [COCO-Stuff 164k]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k&#41;)

[//]: # (- [x] [CHASE_DB1]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#chase-db1&#41;)

[//]: # (- [x] [DRIVE]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#drive&#41;)

[//]: # (- [x] [HRF]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#hrf&#41;)

[//]: # (- [x] [STARE]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#stare&#41;)

[//]: # (- [x] [Dark Zurich]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#dark-zurich&#41;)

[//]: # (- [x] [Nighttime Driving]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#nighttime-driving&#41;)

[//]: # (- [x] [LoveDA]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda&#41;)

[//]: # (- [x] [Potsdam]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-potsdam&#41;)

[//]: # (- [x] [Vaihingen]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-vaihingen&#41;)

[//]: # (- [x] [iSAID]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isaid&#41;)

[//]: # (- [x] [High quality synthetic face occlusion]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#delving-into-high-quality-synthetic-face-occlusion-segmentation-datasets&#41;)

[//]: # (- [x] [ImageNetS]&#40;https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#imagenets&#41;)

[//]: # ()
[//]: # (## FAQ)

[//]: # ()
[//]: # (Please refer to [FAQ]&#40;docs/en/faq.md&#41; for frequently asked questions.)

[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md]&#40;.github/CONTRIBUTING.md&#41; for the contributing guideline.)

[//]: # ()
[//]: # (## Acknowledgement)

[//]: # ()
[//]: # (MMSegmentation is an open source project that welcome any contribution and feedback.)

[//]: # (We wish that the toolbox and benchmark could serve the growing research)

[//]: # (community by providing a flexible as well as standardized toolkit to reimplement existing methods)

[//]: # (and develop their own new semantic segmentation methods.)

[//]: # ()
[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this project useful in your research, please consider cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@misc{mmseg2020,)

[//]: # (    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},)

[//]: # (    author={MMSegmentation Contributors},)

[//]: # (    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},)

[//]: # (    year={2020})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## License)

[//]: # ()
[//]: # (MMSegmentation is released under the Apache 2.0 license, while some specific features in this library are with other licenses. Please refer to [LICENSES.md]&#40;LICENSES.md&#41; for the careful check, if you are using our code for commercial matters.)

[//]: # ()
[//]: # (## Projects in OpenMMLab)

[//]: # ()
[//]: # (- [MMCV]&#40;https://github.com/open-mmlab/mmcv&#41;: OpenMMLab foundational library for computer vision.)

[//]: # (- [MIM]&#40;https://github.com/open-mmlab/mim&#41;: MIM installs OpenMMLab packages.)

[//]: # (- [MMClassification]&#40;https://github.com/open-mmlab/mmclassification&#41;: OpenMMLab image classification toolbox and benchmark.)

[//]: # (- [MMDetection]&#40;https://github.com/open-mmlab/mmdetection&#41;: OpenMMLab detection toolbox and benchmark.)

[//]: # (- [MMDetection3D]&#40;https://github.com/open-mmlab/mmdetection3d&#41;: OpenMMLab's next-generation platform for general 3D object detection.)

[//]: # (- [MMYOLO]&#40;https://github.com/open-mmlab/mmyolo&#41;: OpenMMLab YOLO series toolbox and benchmark.)

[//]: # (- [MMRotate]&#40;https://github.com/open-mmlab/mmrotate&#41;: OpenMMLab rotated object detection toolbox and benchmark.)

[//]: # (- [MMSegmentation]&#40;https://github.com/open-mmlab/mmsegmentation&#41;: OpenMMLab semantic segmentation toolbox and benchmark.)

[//]: # (- [MMOCR]&#40;https://github.com/open-mmlab/mmocr&#41;: OpenMMLab text detection, recognition, and understanding toolbox.)

[//]: # (- [MMPose]&#40;https://github.com/open-mmlab/mmpose&#41;: OpenMMLab pose estimation toolbox and benchmark.)

[//]: # (- [MMHuman3D]&#40;https://github.com/open-mmlab/mmhuman3d&#41;: OpenMMLab 3D human parametric model toolbox and benchmark.)

[//]: # (- [MMSelfSup]&#40;https://github.com/open-mmlab/mmselfsup&#41;: OpenMMLab self-supervised learning toolbox and benchmark.)

[//]: # (- [MMRazor]&#40;https://github.com/open-mmlab/mmrazor&#41;: OpenMMLab model compression toolbox and benchmark.)

[//]: # (- [MMFewShot]&#40;https://github.com/open-mmlab/mmfewshot&#41;: OpenMMLab fewshot learning toolbox and benchmark.)

[//]: # (- [MMAction2]&#40;https://github.com/open-mmlab/mmaction2&#41;: OpenMMLab's next-generation action understanding toolbox and benchmark.)

[//]: # (- [MMTracking]&#40;https://github.com/open-mmlab/mmtracking&#41;: OpenMMLab video perception toolbox and benchmark.)

[//]: # (- [MMFlow]&#40;https://github.com/open-mmlab/mmflow&#41;: OpenMMLab optical flow toolbox and benchmark.)

[//]: # (- [MMEditing]&#40;https://github.com/open-mmlab/mmediting&#41;: OpenMMLab image and video editing toolbox.)

[//]: # (- [MMGeneration]&#40;https://github.com/open-mmlab/mmgeneration&#41;: OpenMMLab image and video generative models toolbox.)

[//]: # (- [MMDeploy]&#40;https://github.com/open-mmlab/mmdeploy&#41;: OpenMMLab Model Deployment Framework.)