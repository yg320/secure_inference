cd /storage/yakir/secure_inference
export PYTHONPATH="${PYTHONPATH}:/storage/yakir/secure_inference"
python /storage/yakir/secure_inference/research/secure_inference_3pc/server.py &
python /storage/yakir/secure_inference/research/secure_inference_3pc/crypto_provider.py &
python /storage/yakir/secure_inference/research/secure_inference_3pc/client.py &
