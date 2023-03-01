
SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py
CHECKPOINT=/home/yakir/iter_20000_0.06.pth
RELU_SPEC_FILE=/home/yakir/assets/resnet_voc/block_spec/0.06.pickle

# With spec
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH  --image_start 0 --image_end 1449 &
#1449


## Baseline
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --secure_config_path $SECURE_CONFIG_PATH &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py  --secure_config_path $SECURE_CONFIG_PATH &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py  --secure_config_path $SECURE_CONFIG_PATH &
