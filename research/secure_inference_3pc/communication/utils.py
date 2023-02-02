import threading

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import TorchSocket, NumpySocket
from research.secure_inference_3pc.timer import timer, Timer
import time
from threading import Thread
import torch
import queue
import numpy as np
import socket

NUMPY_ARR_QUEUE_SIZE = 10

from research.secure_inference_3pc.const import IS_TORCH_BACKEND

if IS_TORCH_BACKEND:
    empty_arr = torch.Tensor([])
else:
    empty_arr = np.array([], dtype=np.int8)

class Receiver(Thread):
    def __init__(self, ip, port, device):
        super(Receiver, self).__init__()
        self.numpy_arr_queue = queue.Queue(maxsize=NUMPY_ARR_QUEUE_SIZE)
        self.ip = ip
        self.port = port

        self.lock = threading.Lock()
        self.stop_running = False
        self.device = device

    def run(self):
        host = "localhost" if self.ip == "localhost" else socket.gethostbyaddr(self.ip)[0]

        SocketClass = TorchSocket if IS_TORCH_BACKEND else NumpySocket
        with SocketClass() as s:
            s.bind((host, self.port))
            s.listen()
            conn, addr = s.accept()

            while conn:
                frame = conn.recv()
                conn.sendall(empty_arr)
                if len(frame) == 0:
                    return
                if IS_TORCH_BACKEND:
                    frame = frame.to(self.device)  # NUMPY_CONVERSION
                self.numpy_arr_queue.put(frame)

    # @timer(name="Receiver.get")
    def get(self):
        arr = self.numpy_arr_queue.get()
        return arr


class Sender(Thread):
    lock = threading.Lock()

    def __init__(self, ip, port, simulated_bandwidth=None):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.ip = ip
        self.port = port
        self.simulated_bandwidth = simulated_bandwidth #100000000  #bits/second
        self.num_of_bytes_sent = 0

    def run(self):

        total_sleep_time = 0
        host = "localhost" if self.ip == "localhost" else socket.gethostbyaddr(self.ip)[0]
        SocketClass = TorchSocket if IS_TORCH_BACKEND else NumpySocket
        with SocketClass() as s:

            connected = False
            while not connected:
                try:
                    s.connect((host, self.port))
                    connected = True
                except ConnectionRefusedError:
                    time.sleep(0.01)

            while True:
                data = self.numpy_arr_queue.get()
                if data is None:
                    s.close()
                    break
                if type(data) == torch.Tensor:
                    data = data.cpu()  # NUMPY_CONVERSION
                    arr_size_bytes = data.numel() * data.element_size()
                else:

                    arr_size_bytes = data.size * data.itemsize
                # assert arr_size_bytes >= 64
                self.num_of_bytes_sent += arr_size_bytes

                if self.simulated_bandwidth:
                    arr_size_bits = 8 * arr_size_bytes
                    send_time = arr_size_bits / self.simulated_bandwidth
                    Sender.lock.acquire()
                    total_sleep_time += send_time
                    time.sleep(send_time)
                    s.sendall(data)
                    s.recv()
                    Sender.lock.release()
                else:
                    # with Timer(name="s.sendall(data)"):
                    s.sendall(data)
                    s.recv()

    def put(self, arr):
        # TODO: why is this copy needed (related to the monster threading bug)
        if arr is not None:
            if not IS_TORCH_BACKEND:
                arr = arr.copy()
        self.numpy_arr_queue.put(arr)

