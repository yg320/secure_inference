#!/usr/bin/python3

import logging
import time

from research.secure_inference_3pc.communication.numpy_socket.numpysocket.numpysocket import NumpySocket
import numpy as np
logger = logging.getLogger('simple server')
logger.setLevel(logging.INFO)

with NumpySocket() as s:
    s.bind(('', 11111))
    s.listen()
    conn, addr = s.accept()
    with conn:
        logger.info(f"connected: {addr}")
        frame = conn.recv()
        conn.sendall(np.array([]))

        print(time.time())
        print(frame)

    logger.info(f"disconnected: {addr}")

