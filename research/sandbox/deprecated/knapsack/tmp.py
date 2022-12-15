import pickle

out = {
    **pickle.load(
        file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_0_0.0833.pickle",
                  'rb')),
    **pickle.load(
        file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_1_0.0833.pickle",
                  'rb')),
    **pickle.load(
        file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_2_0.0833.pickle",
                  'rb')),
    **pickle.load(
        file=open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_3_0.0833.pickle",
                  'rb')),
    # **{'decode_0':[[1,1]]*512}
}

pickle.dump(out, open(f"/home/yakir/Data2/block_relu_specs/deeplabv3_m-v2-d8_256x256_160k_ade20k_iter_0123_0.0833.pickle", 'wb'))