import torch


# TODO: replace redundancy in code, Transfer shapes and not tensors
def get_output_shape(A, B, padding, dilation, stride):
    if type(A) in [tuple, torch.Size]:
        nb_rows_in = A[2]
        nb_cols_in = A[3]
    else:
        nb_rows_in = A.shape[2]
        nb_cols_in = A.shape[3]
    if type(B) in [tuple, torch.Size]:
        nb_rows_kernel = B[2]
        nb_cols_kernel = B[3]
        channel_out = B[0]
    else:
        nb_rows_kernel = B.shape[2]
        nb_cols_kernel = B.shape[3]
        channel_out = B.shape[0]
    nb_rows_out = int(
        ((nb_rows_in + 2 * padding[0] - dilation[0] * (nb_rows_kernel - 1) - 1) / stride[0]) + 1
    )
    nb_cols_out = int(
        ((nb_cols_in + 2 * padding[1] - dilation[1] * (nb_cols_kernel - 1) - 1) / stride[1]) + 1
    )

    return 1, channel_out, nb_rows_out, nb_cols_out

