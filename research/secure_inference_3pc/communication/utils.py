import threading

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket
import time
from threading import Thread
import torch
import queue
import numpy as np
import os

NUMPY_ARR_QUEUE_SIZE = 50

OUT_DIR = "/home/yakir/ports"

class Receiver(Thread):
    def __init__(self, port):
        super(Receiver, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.out_dir = os.path.join(OUT_DIR, str(self.port))

    def run(self):

        while True:
            arr = self.get_data()
            self.numpy_arr_queue.put(arr)

    def get_data(self):
        while not os.path.exists(os.path.join(self.out_dir, "done.npy")):
            time.sleep(0.05)
        arr = np.load(os.path.join(self.out_dir, "arr.npy"))
        os.remove(os.path.join(self.out_dir, "arr.npy"))
        os.remove(os.path.join(self.out_dir, "done.npy"))
        return arr

    def get(self):
        return self.numpy_arr_queue.get()


class Sender(Thread):

    def __init__(self, port, simulated_bandwidth=None):
        super(Sender, self).__init__()
        self.numpy_arr_queue = queue.Queue()
        self.port = port
        self.simulated_bandwidth = simulated_bandwidth #100000000  #bits/second
        self.num_of_bytes_sent = 0
        self.out_dir = os.path.join(OUT_DIR, str(self.port))
        os.makedirs(self.out_dir, exist_ok=True)
        
    def run(self):
        while True:
            data = self.numpy_arr_queue.get()
            self.put_data(data)

    def put_data(self, data):
        if data is None:
            return
        if type(data) == torch.Tensor:
            data = data.numpy()
        while os.path.exists(os.path.join(self.out_dir, "done.npy")):
            time.sleep(0.05)
        np.save(file=os.path.join(self.out_dir, "arr.npy"), arr=data)
        np.save(file=os.path.join(self.out_dir, "done.npy"), arr=[])

    def put(self, data):
        self.put_data(data)