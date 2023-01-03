# import numpy as np
# import time
# import torch
# from research.secure_inference_3pc.timer import Timer
# NUM_BITS = 64
# UNSIGNED_DTYPE = np.uint64
# NUM_OF_COMPARE_BITS = 64
#
# power_numpy = np.arange(NUM_BITS, dtype=UNSIGNED_DTYPE)[np.newaxis][:,::-1]
# powers_torch = torch.from_numpy(power_numpy.astype(np.int64)).to("cuda:0")
#
#
# def decompose(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#     r_shift = value >> power_numpy[:,NUM_BITS - NUM_OF_COMPARE_BITS:]
#     value_bits = np.zeros(shape=(value.shape[0], NUM_OF_COMPARE_BITS), dtype=np.int8)
#     np.bitwise_and(r_shift, np.int8(1), out=value_bits)
#     ret =  value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret
#
# def decompose_torch(value):
#     orig_shape = list(value.shape)
#     value = value.reshape(-1, 1)
#
#     r_shift = value >> powers_torch[:, NUM_BITS - NUM_OF_COMPARE_BITS:]
#     value_bits = r_shift & 1
#
#     ret =  value_bits.reshape(orig_shape + [NUM_OF_COMPARE_BITS])
#     return ret
#
# value = np.random.randint(low=np.iinfo(np.uint64).min, high=np.iinfo(np.uint64).max, size=(1000000,), dtype=np.uint64)
#
# with Timer("decompose - torch"):
#     torch_tensor = torch.from_numpy(value.astype(np.int64)).to("cuda:0")
#     out = decompose_torch(torch_tensor)
#     out = out.to("cpu").numpy().astype(np.uint64)
#
# with Timer("decompose - numpy"):
#     decompose(value)


bits = decompose(r)
c_bits_0 = get_c_party_0(x_bits_0, bits, beta, np.int8(0))
np.multiply(s, c_bits_0, out=s)
with Timer("module_67(s)"):
    d_bits_0 = module_67(s)