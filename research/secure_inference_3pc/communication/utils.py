import threading

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch
import queue
import numpy as np

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

            while conn:
                frame = conn.recv()
                conn.sendall(np.array([]))
                if len(frame) == 0:
                    return
                self.numpy_arr_queue.put(frame)

    def get(self):
        t0 = time.time()
        arr = self.numpy_arr_queue.get()
        t1 = time.time()
        self.get_total_time += (t1-t0)
        return arr


# Client
class Sender(Thread):
    def __init__(self, port):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.simulated_bandwidth = None #100000000  #bits/second

    def run(self):
        num_bytes_send = 0

        with NumpySocket() as s:

            connected = False
            while not connected:
                try:
                    s.connect(("localhost", self.port))
                    connected = True
                except ConnectionRefusedError:
                    time.sleep(0.01)

            while True:
                data = self.numpy_arr_queue.get()
                if data is None:
                    s.close()
                    break
                if type(data) == torch.Tensor:
                    data = data.numpy()

                arr_size_bytes = data.size * data.itemsize

                num_bytes_send += arr_size_bytes

                if self.simulated_bandwidth:
                    arr_size_bits = 8 * arr_size_bytes
                    send_time = arr_size_bits / self.simulated_bandwidth
                    time.sleep(send_time)

                s.sendall(data)
                s.recv(1)

        print(num_bytes_send)
    def put(self, arr):
        self.numpy_arr_queue.put(arr)

