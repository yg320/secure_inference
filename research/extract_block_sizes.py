import os
import argparse
import subprocess

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

os.chdir(os.path.dirname(os.path.dirname(__file__)))
distortion_output_path = os.path.join(args.output_path, "distortion_row")

for gpu_index in range(args.num_gpus):
    # TODO: make it multiprocess (mp.Process), instead of using subprocess. Alternatively, consider using
    #  torch.nn.parallel.parallel_apply, and then, we can discard DistortionCollector
    python_command = \
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
