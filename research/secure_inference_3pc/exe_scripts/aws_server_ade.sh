RELU_SPEC_FILE=/home/ubuntu/specs/ade/0.12.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu_secure_aspp.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --secure_config_path $SECURE_CONFIG_PATH --relu_spec_file $RELU_SPEC_FILE  &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --secure_config_path $SECURE_CONFIG_PATH  &

