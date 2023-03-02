RELU_SPEC_FILE=/home/ubuntu/specs/voc/0.15.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_avg_pool_secure_aspp.py
#SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/segmentation/deeplabv3/deeplabv3_r50-d8_512x512_40k_voc12aug_max_pool_secure_aspp.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --secure_config_path $SECURE_CONFIG_PATH --relu_spec_file $RELU_SPEC_FILE --dummy_image --image_start 0 --image_end 4 &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --secure_config_path $SECURE_CONFIG_PATH --dummy_image --image_start 0 --image_end 4 &
