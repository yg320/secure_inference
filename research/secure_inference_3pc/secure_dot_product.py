
# # Server
# def conv2d(activation_share, weight_share):
#     assert weight_share.shape[2] == weight_share.shape[3]
#     assert activation_share.shape[2] == activation_share.shape[3]
#
#     b, i, m, _ = activation_share.shape
#     output_shape = (b, o, m, m)
#     padding = weight_share.shape[2]
#
#     A_share = get_prf_share(activation_share.shape)
#     B_share = get_prf_share(weight_share.shape)
#     C_share = get_prf_share(output_shape)
#
#
# i = 3
# o = 64
# f = 7
# m = 256
# padding = (f - 1) // 2
# X0 = torch.rand(size=(1, i, m, m))
# X1 = torch.rand(size=(1, i, m, m))
#
# Y0 = torch.rand(size=(o,i, f, f))
# Y1 = torch.rand(size=(o,i, f, f))
#
# X = X0 + X1
# Y = Y0 + Y1
#
#
# A0 = torch.rand(size=(1, i, m, m))
# A1 = torch.rand(size=(1, i, m, m))
#
# B0 = torch.rand(size=(o, i, f, f))
# B1 = torch.rand(size=(o, i, f, f))
#
# C0 = torch.rand(size=(1, o, m, m))
#
# A = A0 + A1
# B = B0 + B1
#
# # A_prime, B_prime, *_ = pre_conv(A, B, padding=3)
# # C1 = post_conv(bias=None, res=A_prime @ B_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) - C0
# C1 = torch.conv2d(A, B, bias=None, stride=1, padding=padding, dilation=1, groups=1) - C0
#
# E0 = X0 - A0
# E1 = X1 - A1
# F0 = Y0 - B0
# F1 = Y1 - B1
#
# E = E0 + E1
# F = F0 + F1
#
#
# # E_prime, F_prime, *_ = pre_conv(E, F, padding=3)
# # X0_prime, Y0_prime, *_ = pre_conv(X0, Y0, padding=3)
# # X1_prime, Y1_prime, *_ = pre_conv(X1, Y1, padding=3)
# #
# # Z0 = post_conv(bias=None, res= 0 * E_prime @ F_prime + X0_prime @ F_prime + E_prime @ Y0_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) + C0
# # Z1 = post_conv(bias=None, res=-1 * E_prime @ F_prime + X1_prime @ F_prime + E_prime @ Y1_prime, batch_size=1, nb_channels_out=1, nb_rows_out=256, nb_cols_out=256) + C1
#
# Z0 = 0 * torch.conv2d(E, F,  bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
#          torch.conv2d(X0, F, bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
#          torch.conv2d(E, Y0, bias=None, stride=1, padding=padding, dilation=1, groups=1) + C0
#
# Z1 = -1 * torch.conv2d(E, F,  bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
#           torch.conv2d(X1, F, bias=None, stride=1, padding=padding, dilation=1, groups=1) + \
#           torch.conv2d(E, Y1, bias=None, stride=1, padding=padding, dilation=1, groups=1) + C1
#
# Z = Z0 + Z1
# Z_normal = torch.conv2d(X, Y, bias=None, stride=1, padding=padding, dilation=1, groups=1)
# print('fds')