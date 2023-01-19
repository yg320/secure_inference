#!/usr/bin/env python3

import socket
import logging
import numpy as np
from io import BytesIO
import torch

from research.secure_inference_3pc.timer import Timer
class NumpySocket(socket.socket):
    def sendall(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("input frame is not a valid numpy array") # should this just call super intead?

        out = self.__pack_frame(frame)
        super().sendall(out)
        logging.debug("frame sent")


    def recv(self, bufsize=65536):

        length_str = super().recv(12)

        if len(length_str) == 0:
            return np.array([])
        else:
            length = int(length_str)
            frameBuffer = bytearray()

            while len(frameBuffer) < length:
                data = super().recv(length - len(frameBuffer))
                frameBuffer += data

            return np.load(BytesIO(frameBuffer), allow_pickle=False)

    def accept(self):
        fd, addr = super()._accept()
        sock = NumpySocket(super().family, super().type, super().proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr


    @staticmethod
    def __pack_frame(frame):
        # out_0 = frame.tobytes()
        # out_0 = len(out_0).to_bytes(12, byteorder='big') + out_0
        # return out_0
        f = BytesIO()
        np.save(f, arr=frame)

        packet_size = str(len(f.getvalue())).zfill(12)
        header = packet_size
        header = bytes(header.encode())  # prepend length of array
        out = bytearray(header)
        # out += header

        f.seek(0)
        out += f.read()
        return out



class TorchSocket(socket.socket):
    def sendall(self, frame):
        if not isinstance(frame, torch.Tensor):
            raise TypeError("input frame is not a valid numpy array")  # should this just call super intead?

        out = self.__pack_frame(frame)
        super().sendall(out)
        logging.debug("frame sent")


    def recv(self, bufsize=4096):

        data = super().recv(bufsize)

        if len(data) == 0:
            return torch.Tensor([])
        else:
            frameBuffer = bytearray()
            length_str, ignored, data = data.partition(b':')
            length = int(length_str)

            frameBuffer += data

            while len(frameBuffer) < length:
                data = super().recv(bufsize)
                frameBuffer += data

        frame = torch.load(BytesIO(frameBuffer))
        return frame

    def accept(self):
        fd, addr = super()._accept()
        sock = TorchSocket(super().family, super().type, super().proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr


    @staticmethod
    def __pack_frame(frame):
        f = BytesIO()
        torch.save(frame, f)

        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        return out
