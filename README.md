## What does this package offer?:

## Enviorment Setup:

## Training:
###  [Segmentation, MobileNetV2, ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py)
- **First, we extract disotrtion for each channel and each block size by running:** 
    - python research/extract_block_sizes.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py --checkpoint {PATH_TO_MMLAB_MODELS}/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --output_path {WORK_DIR}/segmentation --num_samples NUM_SAMPLES --num_gpus NUM_GPUS 
    - we used NUM_SAMPLES=48 over NUM_GPUS=4
- **Now we are ready to get the knapsack optimal patch-sizes by running: (Here we use 6% DReLU budget)**
  - export PYTHONPATH=. ; python research/distortion/knapsack/knapsack_patch_size_extractor.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py --block_size_spec_file_name benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle --channel_distortion_path /storage/yakir/secure_inference/benchmark/segmentation/mobilenet_ade/distortion/distortion_collected/ --ratio 0.06
- **Finally, we can train the network**
  - export PYTHONPATH=. ; bash ./research/mmlab_tools/segmentation/dist_train.sh research/configs/segmentation/mobilenet_v2/deeplabev2_mobilenet_ade20k_finetune.py 4 --load-from mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --work-dir benchmark/segmentation/mobilenet_ade/experiments/0.06 --relu-spec-file benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle


###  [Segmentation, ResNet50, Pascal VOC 2012](https://github.com/open-mmlab/mmsegmentation/blob/master/configs/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug.py)
- **First, we extract disotrtion for each channel and each block size by running:** 
    - python research/extract_block_sizes.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py --checkpoint {PATH_TO_MMLAB_MODELS}/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --output_path {WORK_DIR}/segmentation --num_samples NUM_SAMPLES --num_gpus NUM_GPUS 
    - we used NUM_SAMPLES=48 over NUM_GPUS=4
- **Now we are ready to get the knapsack optimal patch-sizes by running: (Here we use 6% DReLU budget)**
  - export PYTHONPATH=. ; python research/distortion/knapsack/knapsack_patch_size_extractor.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py --block_size_spec_file_name benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle --channel_distortion_path /storage/yakir/secure_inference/benchmark/segmentation/mobilenet_ade/distortion/distortion_collected/ --ratio 0.06
- **Finally, we can train the network**
  - export PYTHONPATH=. ; bash ./research/mmlab_tools/segmentation/dist_train.sh research/configs/segmentation/mobilenet_v2/deeplabev2_mobilenet_ade20k_finetune.py 4 --load-from mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --work-dir benchmark/segmentation/mobilenet_ade/experiments/0.06 --relu-spec-file benchmark/segmentation/mobilenet_ade/distortion/block_spec/0.06.pickle



###  [Classification, ResNet50, ImageNet](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py)

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

## Secure Inference

## Next Steps:
- **Knowledge distillation (similar to DeepReDuce) with out bReLU layer and Knapsack algorithm**
- **Knapsack as a starting point for some iterative algorithm (such as simulated annealing)**
- **For larger networks, where it takes too much time to extract distortion, we can measure distortion in some middle layer and normalize by the appropriate signal to get a SNR measure, this is already supported, and a PoC has been made**
- **Iterative Knapsack, where we work on a bunch of layers each time**

