import threading

from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch
import queue
import numpy as np

def recv(socket):
    print("recv")
    with NumpySocket() as s:
        # print("Binding To Socket")
        s.bind(('', socket))
        # print("Listening...")
        s.listen()
        # print("Accepting...")
        conn, addr = s.accept()
        # print("Receiving...")
        with conn:
            frame = conn.recv()
        # print("Done!")
    print("done recv")

    return frame
    return torch.from_numpy(frame)


def send(socket, data):
    print("send")
    if type(data) == torch.Tensor:
        data = data.numpy()
    with NumpySocket() as s:
        while True:
            succeeded = True
            try:
                s.connect(("localhost", socket))
            except ConnectionRefusedError:
                # print("connection close - sleeping...")
                succeeded = False
                time.sleep(0.03)
            if succeeded:
                break
        # print("Connecting! Sending data")
        s.sendall(data)
    time.sleep(0.03)
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


# Server
class Receiver(Thread):
    def __init__(self, port):
        super(Receiver, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port

        self.lock = threading.Lock()
        self.stop_running = False
        self.get_total_time = 0

    def run(self):

        with NumpySocket() as s:
            s.bind(('', self.port))
            s.listen()
            conn, addr = s.accept()
            print(f"Accepted Connection port = {self.port}")

            while conn:
                print(f"Recv from {self.port}")
                frame = conn.recv()
                conn.sendall(np.array([]))
                if len(frame) == 0:
                    return
                print(f"Done Recv from {self.port}, {frame.flatten()[0]}, {frame.shape}")

                self.numpy_arr_queue.put(frame)
            print("Connection Done")

    def get(self):
        t0 = time.time()
        arr = self.numpy_arr_queue.get()
        t1 = time.time()
        self.get_total_time += (t1-t0)
        # print("====================")
        # print("get: ", self.get_total_time)
        return arr


# Client
class Sender(Thread):
    def __init__(self, port):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.simulated_bandwidth = None #100000000  #bits/second
        self.put_tot_time = 0
        self.connected = False

    def run(self):

        with NumpySocket() as s:
            connected = False
            while not connected:
                try:
                    s.connect(("localhost", self.port))
                    connected = True
                except ConnectionRefusedError:
                    time.sleep(0.01)
            print(f"Connected to port = {self.port}")
            while True:
                data = self.numpy_arr_queue.get()
                if data is None:
                    s.close()
                    break
                if type(data) == torch.Tensor:
                    data = data.numpy()
                print(f"Sending data to port = {self.port}, {data.flatten()[0]} {data.shape}")
                s.sendall(data)
                s.recv(1)
                print(f"Done Sending data to port = {self.port}")
            print("Connection Done")

    def put(self, arr):
        t0 = time.time()
        self.numpy_arr_queue.put(arr)
        t1 = time.time()
        self.put_tot_time += (t1-t0)


if __name__ == "__main__":
    receiver = Receiver(socket=12332)
    receiver.start()
    receiver.numpy_arr_queue.get()
    print('f')

    import numpy as np
    send_v2(12332, np.arange(10000))