import pickle
import os
import glob
import numpy as np

input_path = ""
out_path = ""
layer_names = ["layer1_1_1"]

for layer_name in layer_names:
    glob_pattern = os.path.join(input_path, f"{layer_name}_*.pickle")
    files = glob.glob(glob_pattern)
    assert len(files) == 8, glob_pattern

    noise = np.stack([pickle.load(open(f, 'rb'))["Noise"] for f in files])
    noise = noise.mean(axis=0).T

    # signal = np.stack([pickle.load(open(f, 'rb'))["Signal"] for f in files])
    # signal = signal.mean(axis=0).T

    loss = np.stack([pickle.load(open(f, 'rb'))["Distorted Loss"] for f in files])
    loss = loss.mean(axis=0).T

    distortion = noise
    distortion = distortion[:-1]
    tt = -np.array(distortion).T

    loss = loss[:-1].T

    np.save(os.path.join(out_path, "loss", f"{layer_name}.npy"), tt)
    np.save(os.path.join(out_path, "distortion", f"{layer_name}.npy"), loss)

