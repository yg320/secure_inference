
CHECKPOINT=models/ratio_0.15.pth
RELU_SPEC_FILE=block_spec/0.15.pickle
SECURE_CONFIG_PATH=research/configs/classification/resnet/resnet50_in1k/resnet50_in1k_avg_pool.py


export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH  --image_start 0 --image_end 5 &
