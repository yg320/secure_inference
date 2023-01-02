#!/usr/bin/python3

import logging
import numpy as np
from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket
import socket
import time

logger = logging.getLogger('simple client')
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.connect(("localhost", 11111))
    
    logger.info("sending numpy array:")
    frame = np.random.random((1000, ))
    print(time.time())
    s.sendall(frame)
    s.recv(10)

