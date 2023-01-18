cd /home/yakir/PycharmProjects/secure_inference
export PYTHONPATH="${PYTHONPATH}:/home/yakir/PycharmProjects/secure_inference"
/home/yakir/anaconda3/envs/open-mmlab-mmseg-numba/bin/python /home/yakir/PycharmProjects/secure_inference/research/secure_inference_3pc/server.py  > /dev/null 2>&1 &
/home/yakir/anaconda3/envs/open-mmlab-mmseg-numba/bin/python /home/yakir/PycharmProjects/secure_inference/research/secure_inference_3pc/crypto_provider.py &
/home/yakir/anaconda3/envs/open-mmlab-mmseg-numba/bin/python /home/yakir/PycharmProjects/secure_inference/research/secure_inference_3pc/client.py   &
