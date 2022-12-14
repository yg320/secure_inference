#!/usr/bin/python3

from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import numpy as np
import time
import socket

with NumpySocket() as s:
    s.connect(("localhost", 9999))

    for frame_index in range(10):

        frame = np.random.random(size=(np.random.randint(low=1000, high=2000),))
        # time.sleep(0.1)

        try:
            print("sending ", frame_index)
            s.sendall(frame)
            s.recv(1)
            print("done sending ", frame_index)
        except:
            break

    s.close()