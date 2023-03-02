#CHECKPOINT=/home/yakir/assets/resnet_imagenet/models/ratio_0.15.pth
#RELU_SPEC_FILE=/home/yakir/assets/resnet_imagenet/block_spec/0.15.pickle
#SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py

#CHECKPOINT=/home/yakir/PycharmProjects/secure_inference/work_dirs/baseline_0.12/epoch_120.pth
#RELU_SPEC_FILE=/home/yakir/deepreduce_comparison_v3/distortions/baseline/block_sizes/0.12.pickle
#SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/work_dirs/baseline_0.12/baseline_0.12.py

CHECKPOINT=/home/yakir/assets/mobilenet_ade/models/ratio_0.06.pth
RELU_SPEC_FILE=/home/yakir/assets/mobilenet_ade/block_spec/0.06.pickle
SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu_secure_aspp.py

#CHECKPOINT=/home/yakir/assets/resnet_voc/models/iter_20000.pth
#RELU_SPEC_FILE=/home/yakir/assets/resnet_voc/block_spec/0.12.pickle
#SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py
#SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_max_pool_secure_aspp.py

# With spec
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH  --image_start 0 --image_end 2000 --dump_dir /home/yakir/evaluation/0.06/ade20k_lsb_0_msb_40 &
#1449


## Baseline
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --secure_config_path $SECURE_CONFIG_PATH &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py  --secure_config_path $SECURE_CONFIG_PATH &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py  --secure_config_path $SECURE_CONFIG_PATH &
