import os
import argparse
import subprocess
# python research/extract_block_sizes.py --config research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k.py --checkpoint mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth --output_path /storage/yakir/secure_inference/benchmark/segmentation --num_samples 48 --num_gpus 4
# research/extract_block_sizes.py --config research/configs/classification/resnet/resnet18_2xb64_cifar100_rereduce_params.py --checkpoint /home/yakir/PycharmProjects/secure_inference/work_dirs/resnet18_2xb64_cifar100/epoch_200.pth --output_path /home/yakir/distortion_200 --num_samples 512 --num_gpus 2
parser = argparse.ArgumentParser(description='')

parser.add_argument('--config', type=str, default="/home/yakir/PycharmProjects/secure_inference/research/configs/segmentation/mobilenet_v2/deeplabv3_m-v2-d8_512x512_160k_ade20k_relu.py")
parser.add_argument('--checkpoint', type=str, default="/home/yakir/PycharmProjects/secure_inference/mmlab_models/segmentation/deeplabv3_m-v2-d8_512x512_160k_ade20k_20200825_223255-63986343.pth")
parser.add_argument('--output_path', type=str, default="/home/yakir/tmptmptmp2")
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--num_gpus', type=int, default=2)

args = parser.parse_args()

distortion_extractor_script = os.path.join(os.path.dirname(__file__), "distortion", "distortion_extractor.py")
batch_size = args.num_samples // args.num_gpus

distortion_extraction_processes = []

chdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(chdir)
os.chdir(chdir)
distortion_output_path = os.path.join(args.output_path, "distortion_row")
PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:' + chdir + '"; '


for gpu_index in range(args.num_gpus):
    # TODO: make it multiprocess (mp.Process), instead of using subprocess. Alternatively, consider using
    #  torch.nn.parallel.parallel_apply, and then, we can discard DistortionCollector
    python_command = \
        PYTHON_PATH_EXPORT + \
        f'python {distortion_extractor_script} ' + \
        f'--config {args.config} ' + \
        f'--checkpoint {args.checkpoint} ' + \
        f'--output_path {distortion_output_path} ' + \
        f'--batch_size {batch_size} ' + \
        f'--batch_index {gpu_index} ' + \
        f'--gpu_id {gpu_index} '
    distortion_extraction_processes.append(
        subprocess.Popen(python_command, shell=True)
    )

exit_codes = [p.wait() for p in distortion_extraction_processes]
#
# DistortionCollector(output_path=distortion_output_path).collect()
