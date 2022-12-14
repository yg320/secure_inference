import threading

from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch
import queue

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
            print(f"Listening To Port {self.port} at time {time.time()}")
            while not self.is_done():
                conn, addr = s.accept()

                try:
                    with conn:
                        frame = conn.recv()
                    self.numpy_arr_queue.put(frame)
                finally:
                    conn.close()
        print("Quit Receiver")

    def get(self):
        t0 = time.time()
        arr = self.numpy_arr_queue.get()
        t1 = time.time()
        self.get_total_time += (t1-t0)
        # print("====================")
        # print("get: ", self.get_total_time)
        return arr

    def is_done(self):
        with self.lock:
            return self.stop_running

    def make_stop(self):
        with self.lock:
            self.stop_running = True
# class Sender:
#     def __init__(self, socket):
#         self.numpy_arr_queue = queue.Queue()
#         self.socket = socket
#
#     def put(self, arr):
#         if type(arr) == torch.Tensor:
#             arr = arr.numpy()
#
#         with NumpySocket() as s:
#             s.connect(("localhost", self.socket))
#             s.sendall(arr)
#
#     def start(self):
#         pass
class Sender(Thread):
    def __init__(self, port):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.lock = threading.Lock()
        self.stop_running = False
        self.simulated_bandwidth = None #100000000  #bits/second
        self.put_tot_time = 0
        self.connected = False

    def run(self):
        # with NumpySocket() as s:
        #     while not self.connected:
        #         self.connected = True
        #         try:
        #             s.connect(("localhost", self.port))
        #         except ConnectionRefusedError:
        #             time.sleep(0.01)
        #             self.connected = False
        #
        #     while True:
        #         data = self.numpy_arr_queue.get()
        #         if type(data) == torch.Tensor:
        #             data = data.numpy()
        #         print(f"Sending Data to port - {self.port} at time {time.time()}")
        #
        #         data_sent = False
        #         while not data_sent:
        #             try:
        #                 s.sendall(data)
        #                 data_sent = True
        #             except BrokenPipeError:
        #                 time.sleep(0.1)


        while True:
            data = self.numpy_arr_queue.get()
            if type(data) == torch.Tensor:
                data = data.numpy()
            with NumpySocket() as s:
                if self.simulated_bandwidth:
                    arr_size_bits = (data.size * data.itemsize) * 8
                    send_time = arr_size_bits / self.simulated_bandwidth
                    time.sleep(send_time)

                connected = False
                while not connected:
                    try:
                        s.connect(("localhost", self.port))
                        connected = True
                    except ConnectionRefusedError:
                        time.sleep(0.01)

                s.sendall(data)
        print("Quit Sender")

    def put(self, arr):
        t0 = time.time()
        self.numpy_arr_queue.put(arr)
        t1 = time.time()
        self.put_tot_time += (t1-t0)
        # print("====================")
        # print("put: ", self.put_tot_time)

    def is_done(self):
        with self.lock:
            return self.stop_running

    def make_stop(self):
        with self.lock:
            self.stop_running = True

# def send_v2(socket, data):
#
#     if type(data) == torch.Tensor:
#         data = data.numpy()
#
#     with NumpySocket() as s:
#         s.connect(("localhost", socket))
#         s.sendall(data)


if __name__ == "__main__":
    receiver = Receiver(socket=12332)
    receiver.start()
    receiver.numpy_arr_queue.get()
    print('f')

    import numpy as np
    send_v2(12332, np.arange(10000))