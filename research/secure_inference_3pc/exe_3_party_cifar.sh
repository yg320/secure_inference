
CHECKPOINT=/home/yakir/PycharmProjects/secure_inference/work_dirs/benchmark/baseline_0.18/epoch_120.pth
RELU_SPEC_FILE=/home/yakir/deepreduce_comparison_v3/distortions/baseline/block_sizes/0.18.pickle
SECURE_CONFIG_PATH=/home/yakir/PycharmProjects/secure_inference/work_dirs/benchmark/baseline_0.18/baseline_0.18.py


export PYTHONPATH="." ; python research/secure_inference_3pc/parties/server/run.py --model_path $CHECKPOINT --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/crypto_provider/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH &
export PYTHONPATH="." ; python research/secure_inference_3pc/parties/client/run.py --relu_spec_file $RELU_SPEC_FILE --secure_config_path $SECURE_CONFIG_PATH  --image_start 0 --image_end 5 &
