
CHECKPOINT=/storage/yakir/secure_inference/benchmark/segmentation/resnet_voc/experiments/0.06/iter_20000.pth
RELU_SPEC_FILE=/storage/yakir/secure_inference/benchmark/segmentation/resnet_voc/distortion/block_spec/0.06.pickle
SECURE_CONFIG_PATH=research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py

# With spec
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH  --image_start 0 --image_end 1449 --skip_existing --dump_dir /storage/yakir/evaluation/0.06/voc_18_stoc &
