RELU_SPEC_FILE=/home/ubuntu/specs/imagenet/0.12.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py
#SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_maxpool.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --secure_config_path $SECURE_CONFIG_PATH --relu_spec_file $RELU_SPEC_FILE --dummy_image --image_start 0 --image_end 3 &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --secure_config_path $SECURE_CONFIG_PATH --dummy_image --image_start 0 --image_end 3 &

