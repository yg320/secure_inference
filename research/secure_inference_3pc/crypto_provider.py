import torch
from research.communication.utils import send

class CryptoProvider:
    def __init__(self, server_provider_prf, client_provider_prf, client_provider_socket, provider_client_socket,
                 server_provider_socket, provider_server_socket):
        self.server_provider_prf = server_provider_prf
        self.client_provider_prf = client_provider_prf

        self.client_provider_socket = client_provider_socket
        self.provider_client_socket = provider_client_socket

        self.server_provider_socket = server_provider_socket
        self.provider_server_socket = provider_server_socket

    def conv2d(self, X_share_shape, Y_share_shape, stride=1):

        b, i, m, _ = X_share_shape
        m = m // stride
        o, _, _, f = Y_share_shape
        output_shape = (b, o, m, m)
        padding = (f - 1) // 2
        A_share_server = torch.rand(size=X_share_shape, generator=self.server_provider_prf)
        B_share_server = torch.rand(size=Y_share_shape, generator=self.server_provider_prf)
        C_share_server = torch.rand(size=output_shape, generator=self.server_provider_prf)

        A_share_client = torch.rand(size=X_share_shape, generator=self.client_provider_prf)
        B_share_client = torch.rand(size=Y_share_shape, generator=self.client_provider_prf)

        A = A_share_client + A_share_server
        B = B_share_client + B_share_server

        C_share_client = torch.conv2d(A, B, bias=None, stride=stride, padding=padding, dilation=1, groups=1) - C_share_server
        send(socket=self.provider_client_socket, data=C_share_client)

        print("HJEY")

if __name__ == "__main__":

    crypt_provider = CryptoProvider(
        server_provider_prf=torch.Generator().manual_seed(3),
        client_provider_prf=torch.Generator().manual_seed(2),
        client_provider_socket=22125,
        provider_client_socket=22126,
        server_provider_socket=22127,
        provider_server_socket=22128
    )

    crypt_provider.conv2d((1, 3, 256, 256), (32, 3, 3, 3), stride=2)
    crypt_provider.conv2d((1, 32, 128, 128), (32, 32, 3, 3))
    crypt_provider.conv2d((1, 32, 128, 128), (64, 32, 3, 3))

    print('fdslkj')
