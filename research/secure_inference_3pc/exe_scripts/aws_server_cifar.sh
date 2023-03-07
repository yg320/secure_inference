RELU_SPEC_FILE=/home/ubuntu/specs/cifar/0.15.pickle
SECURE_CONFIG_PATH=/home/ubuntu/secure_inference/research/configs/classification/resnet/resnet18_cifar100/baseline.py

export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --secure_config_path $SECURE_CONFIG_PATH --relu_spec_file $RELU_SPEC_FILE  &
#export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --secure_config_path $SECURE_CONFIG_PATH  &

