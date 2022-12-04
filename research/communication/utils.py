from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch

def recv(socket):
    with NumpySocket() as s:
        print("Binding To Socket")
        s.bind(('', socket))
        print("Listening...")
        s.listen()
        print("Accepting...")
        conn, addr = s.accept()
        print("Receiving...")
        with conn:
            frame = conn.recv()
        print("Done!")
    return torch.from_numpy(frame)


def send(socket, data):
    data = data.numpy()
    with NumpySocket() as s:
        while True:
            succeeded = True
            try:
                s.connect(("localhost", socket))
            except ConnectionRefusedError:
                print("connection close - sleeping...")
                succeeded = False
                time.sleep(0.01)
            if succeeded:
                break
        print("Connecting! Sending data")
        s.sendall(data)
    print("Done Sending!")

def recv_wrapper(socket, out):
    out[0] = recv(socket)


def send_recv(socket_a, socket_b, data):

    recv_list = [None]

    t0 = Thread(target=recv_wrapper, args=(socket_a, recv_list))
    t1 = Thread(target=send, args=(socket_b, data))

    t0.start()
    t1.start()

    t0.join()
    t1.join()

    return recv_list[0]

