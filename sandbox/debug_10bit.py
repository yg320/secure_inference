import numpy as np

bit = 40
real_msb = mean_tensors & 2 ** bit >> bit  # LSB-24
mask_high = np.uint64(sum(2 ** i for i in range(bit, 64))).astype(np.int64)
mask_low = np.uint64(sum(2 ** i for i in range(0, bit))).astype(np.int64)
mask = mask_high * real_msb + mask_low
to_use = mask & mean_tensors


client_share = np.int64([-1771766758641243084,  7358086221102220783, -4408478206965873863, -631915508614433194,  1211068313044917935,  8806390471207033043, -6064990067284593181,  7922886145571108739,  3176049059518718522, 5316551296872348931])
server_share = np.int64([ 1771766758641305368, -7358086221102240419,  4408478206965869934, 631915508614452980, -1211068313044949388, -8806390471206944353, 6064990067284595297, -7922886145571112566, -3176049059518759701, -5316551296872390638])
client_share_16bit = (client_share >> 4).astype(np.int16).astype(np.int64)
server_share_16bit = (server_share >> 4).astype(np.int16).astype(np.int64)
client_share_16bit + server_share_16bit > 0
client_share + server_share > 0
np.binary_repr(client_share[0]>>4, width=64)
np.binary_repr(server_share[0], width=64)
np.binary_repr(client_share_16bit[0], width=64)
np.binary_repr(server_share_16bit[0], width=64)

client_share_16bit + server_share_16bit

client_share + server_share