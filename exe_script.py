import os

os.system('export PYTHONPATH=\"${PYTHONPATH}:/storage/yakir/secure_inference\"; '
          'bash ./research/pipeline/dist_train.sh research/pipeline/configs/m-v2_256x256_ade20k/relu_spec_0.25.py --load-from ./iter_160000.pth')
