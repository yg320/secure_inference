import torch
from research.communication.utils import send_recv, recv, send


class Client:
    def __init__(self, client_server_prf, client_provider_prf, client_server_socket, server_client_socket, client_provider_socket, provider_client_socket):
        self.client_server_prf = client_server_prf
        self.client_provider_prf = client_provider_prf
        self.client_server_socket = client_server_socket
        self.server_client_socket = server_client_socket
        self.client_provider_socket = client_provider_socket
        self.provider_client_socket = provider_client_socket

    def conv2d(self, X_share, Y_share, stride=1):
        assert Y_share.shape[2] == Y_share.shape[3]
        assert Y_share.shape[1] == X_share.shape[1]
        assert X_share.shape[2] == X_share.shape[3]

        b, i, m, _ = X_share.shape
        m = m // stride
        o, _, _, f = Y_share.shape
        padding = (f - 1) // 2

        A_share = torch.rand(size=X_share.shape, generator=self.client_provider_prf)
        B_share = torch.rand(size=Y_share.shape, generator=self.client_provider_prf)
        C_share = recv(socket=self.provider_client_socket)

        E_share = X_share - A_share
        F_share = Y_share - B_share

        E_share_server = recv(self.server_client_socket)
        send(self.client_server_socket, E_share)
        F_share_server = recv(self.server_client_socket)
        send(self.client_server_socket, F_share)

        # E_share_server = send_recv(self.server_client_socket, self.client_server_socket, E_share)
        # F_share_server = send_recv(self.server_client_socket, self.client_server_socket, F_share)

        E = E_share_server + E_share
        F = F_share_server + F_share

        return torch.conv2d(X_share, F, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + \
            torch.conv2d(E, Y_share, bias=None, stride=stride, padding=padding, dilation=1, groups=1) + C_share



if __name__ == "__main__":
    # !/usr/bin/python3
    global_prf = torch.Generator().manual_seed(0)
    image = torch.rand(size=(1, 3, 256, 256), generator=global_prf)
    weight_0 = torch.rand(size=(32, 3, 3, 3), generator=global_prf)
    weight_1 = torch.rand(size=(32, 32, 3, 3), generator=global_prf)
    weight_2 = torch.rand(size=(64, 32, 3, 3), generator=global_prf)

    client = Client(
        client_server_prf=torch.Generator().manual_seed(1),
        client_provider_prf=torch.Generator().manual_seed(2),
        client_server_socket=22123,
        server_client_socket=22124,
        client_provider_socket=22125,
        provider_client_socket=22126,
    )

    image_share_server = torch.rand(size=(1, 3, 256, 256), generator=client.client_server_prf)
    weight_0_share_client = torch.rand(size=(32, 3, 3, 3), generator=client.client_server_prf)
    weight_1_share_client = torch.rand(size=(32, 32, 3, 3), generator=client.client_server_prf)
    weight_2_share_client = torch.rand(size=(64, 32, 3, 3), generator=client.client_server_prf)

    image_share_client = image - image_share_server

    activation_share_client = image_share_client
    activation_share_client = client.conv2d(activation_share_client, weight_0_share_client, stride=2)
    activation_share_client = client.conv2d(activation_share_client, weight_1_share_client)
    activation_share_client = client.conv2d(activation_share_client, weight_2_share_client)

    # activation_share_client = client.conv2d(image_share_client, weight_share_client)
    send(socket=client.client_server_socket, data=activation_share_client)

    print('fdslkj')


    # print(send_recv(22123, 11213, image.numpy()))

    # server_client_socket = 22123
    # client_server_socket = 11213
    # recv_list = [None]
    #
    # t0 = Thread(target=recv, args=(client_server_socket,recv_list))
    # t1 = Thread(target=send, args=(server_client_socket,))
    #
    # t0.start()
    # t1.start()
    #
    # t0.join()
    # t1.join()
    #
    # print(recv_list[0])
    # send(ser  ver_client_socket, data=np.arange(1000, 2000))
    # print(recv(client_server_socket))
