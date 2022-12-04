import torch
from research.communication.utils import send_recv, send, recv
import time

class Server:
    def __init__(self, client_server_prf, server_provider_prf, client_server_socket, server_client_socket, server_provider_socket, provider_server_socket):
        self.client_server_prf = client_server_prf
        self.server_provider_prf = server_provider_prf
        self.client_server_socket = client_server_socket
        self.server_client_socket = server_client_socket
        self.server_provider_socket = server_provider_socket
        self.provider_server_socket = provider_server_socket

    def conv2d(self, X_share, Y_share, stride=1):
        assert Y_share.shape[2] == Y_share.shape[3]
        assert Y_share.shape[1] == X_share.shape[1]
        assert X_share.shape[2] == X_share.shape[3]

        b, i, m, _ = X_share.shape
        m = m // stride
        o, _, _, f = Y_share.shape
        output_shape = (b, o, m, m)
        padding = (f - 1) // 2

        A_share = torch.rand(size=X_share.shape, generator=self.server_provider_prf)
        B_share = torch.rand(size=Y_share.shape, generator=self.server_provider_prf)
        C_share = torch.rand(size=output_shape, generator=self.server_provider_prf)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        send(self.server_client_socket, E_share)
        E_share_client = recv(self.client_server_socket)
        send(self.server_client_socket, F_share)
        F_share_client = recv(self.client_server_socket)

        # E_share_client = send_recv(self.client_server_socket, self.server_client_socket, E_share)
        # F_share_client = send_recv(self.client_server_socket, self.server_client_socket, F_share)

        E = E_share_client + E_share
        F = F_share_client + F_share

        return -torch.conv2d(E, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share


if __name__ == "__main__":

    global_prf = torch.Generator().manual_seed(0)
    image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)

    server = Server(
        client_server_prf=torch.Generator().manual_seed(1),
        server_provider_prf=torch.Generator().manual_seed(3),
        client_server_socket=22123,
        server_client_socket=22124,
        server_provider_socket=22127,
        provider_server_socket=22128,
    )

    image_share_server = torch.rand(size=(1, 3, 256, 256), generator=server.client_server_prf)
    weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=server.client_server_prf)
    weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=server.client_server_prf)
    weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=server.client_server_prf)

    weight_0_share_server = weight_0 - weight_0_share_client
    weight_1_share_server = weight_1 - weight_1_share_client
    weight_2_share_server = weight_2 - weight_2_share_client

    t0 = time.time()
    activation_share_server = image_share_server
    activation_share_server = server.conv2d(activation_share_server, weight_0_share_server, stride=2)
    activation_share_server = server.conv2d(activation_share_server, weight_1_share_server)
    activation_share_server = server.conv2d(activation_share_server, weight_2_share_server)
    print(time.time() - t0)
    activation_share_client = recv(server.client_server_socket)
    #
    # activation_recon = activation_share_client + activation_share_server
    t0 = time.time()
    activation_non_secure = torch.conv2d(torch.conv2d(torch.conv2d(image, weight_0, padding=1, stride=2), weight_1, padding=1), weight_2, padding=1)
    print(time.time() - t0)
    print((activation_non_secure - (activation_share_server + activation_share_client)).abs().max())
    print('fdslkj')
    # weights = torch.rand(size=(64, 3, 7, 7))
    # print(send_recv(11213, 22123 , weights.numpy()))
    # server_client_socket = 22123
    # client_server_socket = 11213
    # recv_list = [None]
    # t0 = Thread(target=recv, args=(server_client_socket, recv_list))
    # t1 = Thread(target=send, args=(client_server_socket,))
    #
    # t0.start()
    # t1.start()
    #
    # t0.join()
    # t1.join()
    # print(recv_list[0])

    #
    #
    #
    # print(recv(server_client_socket))
    # send(client_server_socket, data=np.arange(1000))
    # with NumpySocket() as s:
    #     s.bind(('', 9999))
    #     s.listen()
    #     conn, addr = s.accept()
    #     with conn:
    #         frame = conn.recv()
    #
