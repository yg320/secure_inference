import os
import argparse
import subprocess
from research.distortion.distortion_collector import DistortionCollector
parser = argparse.ArgumentParser(description='')

parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--num_gpus', type=int, default=2)
parser.add_argument('--batch_index_start', type=int, default=None)

args = parser.parse_args()

distortion_extractor_script = os.path.join(os.path.dirname(__file__), "distortion", "distortion_extractor.py")
batch_size = args.num_samples // args.num_gpus

distortion_extraction_processes = []

chdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(chdir)
distortion_output_path = os.path.join(args.output_path, "distortion_row")
PYTHON_PATH_EXPORT = 'export PYTHONPATH=\"${PYTHONPATH}:' + chdir + '"; '


for gpu_index in range(args.num_gpus):
    # TODO: make it multiprocess (mp.Process), instead of using subprocess. Alternatively, consider using
    #  torch.nn.parallel.parallel_apply, and then, we can discard DistortionCollector
    if args.batch_index_start is None:
        batch_index = gpu_index
    else:
        batch_index = args.batch_index_start + gpu_index
    python_command = \
        PYTHON_PATH_EXPORT + \
        f'python {distortion_extractor_script} ' + \
        f'--config {args.config} ' + \
        f'--checkpoint {args.checkpoint} ' + \
        f'--output_path {distortion_output_path} ' + \
        f'--batch_size {batch_size} ' + \
        f'--batch_index {batch_index} ' + \
        f'--gpu_id {gpu_index} '
    distortion_extraction_processes.append(
        subprocess.Popen(python_command, shell=True)
    )

exit_codes = [p.wait() for p in distortion_extraction_processes]

DistortionCollector(base_path=args.output_path).collect()
