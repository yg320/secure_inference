import threading

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch
import queue
import numpy as np

NUMPY_ARR_QUEUE_SIZE = 10

class Receiver(Thread):
    def __init__(self, port):
        super(Receiver, self).__init__()
        self.numpy_arr_queue = queue.Queue(maxsize=NUMPY_ARR_QUEUE_SIZE)
        self.port = port

        self.lock = threading.Lock()
        self.stop_running = False

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
        arr = self.numpy_arr_queue.get()
        return arr


class Sender(Thread):
    lock = threading.Lock()

    def __init__(self, port, simulated_bandwidth=None):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.simulated_bandwidth = simulated_bandwidth #100000000  #bits/second
        self.num_of_bytes_sent = 0

    def run(self):

        total_sleep_time = 0

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
                assert arr_size_bytes >= 64
                self.num_of_bytes_sent += arr_size_bytes

                if self.simulated_bandwidth:
                    arr_size_bits = 8 * arr_size_bytes
                    send_time = arr_size_bits / self.simulated_bandwidth
                    Sender.lock.acquire()
                    total_sleep_time += send_time
                    time.sleep(send_time)
                    s.sendall(data)
                    s.recv(10)
                    Sender.lock.release()
                else:
                    s.sendall(data)
                    s.recv(10)

    def put(self, arr):
        # TODO: why is this copy needed (related to the monster threading bug)
        if arr is not None:
            arr = arr.copy()
        self.numpy_arr_queue.put(arr)

