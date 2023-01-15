import pickle
import numpy as np
import os
a = pickle.load(open("relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_0/block_size_spec_0.1.pickle", 'rb'))
b = pickle.load(open("relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_1/block_size_spec_0.1.pickle", 'rb'))

out_d = dict()

for k in a.keys():
    c = np.ones_like(a[k])
    a_mask = (~np.all(a[k] == [1, 1], axis=1))
    b_mask = (~np.all(b[k] == [1, 1], axis=1))
    assert not (a_mask & b_mask).any()
    c[a_mask] = a[k][a_mask]
    c[b_mask] = b[k][b_mask]
    out_d[k] = c

os.makedirs("relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_01", exist_ok=True)
pickle.dump(out_d, open("relu_spec_files/classification/resnet50_8xb32_in1k_iterative/iter_01/block_size_spec_0.1.pickle", 'wb'))