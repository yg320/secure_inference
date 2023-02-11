import glob
import os
import numpy as np
from tqdm import tqdm


class DistortionCollector:
    def __init__(self, base_path):
        self.base_path = base_path
        self.include_zero_channels = False

    def collect(self):
        distortion_row_path = os.path.join(self.base_path, "distortion_row")
        if self.include_zero_channels:
            distortion_collected_path = os.path.join(self.base_path, "distortion_collected_with_zero_channels")
        else:
            distortion_collected_path = os.path.join(self.base_path, "distortion_collected")
        os.makedirs(distortion_collected_path)

        all_files = [os.path.basename(x) for x in glob.glob(os.path.join(distortion_row_path, "*.npy"))]
        layer_names = list(set(["_".join(x.split("_")[:-2]) for x in all_files]))
        num_of_batches = len(glob.glob(os.path.join(distortion_row_path, f"{layer_names[0]}_0_*.npy")))

        for layer_name in tqdm(layer_names):
            num_channels = len(glob.glob(os.path.join(distortion_row_path, f"{layer_name}_*_0.npy")))
            distortions = []
            for channel in range(num_channels):
                files = [os.path.join(distortion_row_path, f"{layer_name}_{channel}_{batch_index}.npy") for batch_index
                         in range(num_of_batches)]
                distortions.append(np.stack([np.load(f) for f in files]))

            distortions = np.array(distortions)
            distortions = distortions.mean(axis=1)

            if not self.include_zero_channels:
                distortions = distortions[:, :-1]

            np.save(os.path.join(distortion_collected_path, f"{layer_name}.npy"), -distortions)

# DistortionCollector("/home/yakir/distortion_200_mini_64").collect()