#!/usr/bin/python3

import logging

from research.numpy_socket_test.numpysocket.numpysocket import NumpySocket
import cv2
import numpy as np
import time
logger = logging.getLogger('OpenCV server')
logger.setLevel(logging.INFO)
frame_index = 0
with NumpySocket() as s:
    s.bind(('', 9999))


    try:
        s.listen()
        conn, addr = s.accept()

        logger.info(f"connected: {addr}")
        while conn:
            frame = conn.recv()
            conn.sendall(np.array([]))
            time.sleep(0.2)

            if len(frame) == 0:
                print("fdsfdsfdsf")
                break
            print(frame_index, frame.shape)
            frame_index += 1
    except ConnectionResetError:
        pass

