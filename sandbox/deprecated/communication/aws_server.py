import socket
import logging
import numpy as np
from io import BytesIO
import time

class NumpySocket(socket.socket):
    def sendall(self, frame):
        if not isinstance(frame, np.ndarray):
            raise TypeError("input frame is not a valid numpy array")  # should this just call super intead?

        out = self.__pack_frame(frame)
        super().sendall(out)
        logging.debug("frame sent")

    def recv(self, bufsize=1024):
        length = None
        frameBuffer = bytearray()
        while True:
            data = super().recv(bufsize)
            if len(data) == 0:
                return np.array([])
            frameBuffer += data
            if len(frameBuffer) == length:
                break
            while True:
                if length is None:
                    if b':' not in frameBuffer:
                        break
                    # remove the length bytes from the front of frameBuffer
                    # leave any remaining bytes in the frameBuffer!
                    length_str, ignored, frameBuffer = frameBuffer.partition(b':')
                    length = int(length_str)
                if len(frameBuffer) < length:
                    break
                # split off the full message from the remaining bytes
                # leave any remaining bytes in the frameBuffer!
                frameBuffer = frameBuffer[length:]
                length = None
                break

        frame = np.load(BytesIO(frameBuffer), allow_pickle=True)['frame']
        logging.debug("frame received")
        return frame

    def accept(self):
        fd, addr = super()._accept()
        sock = NumpySocket(super().family, super().type, super().proto, fileno=fd)

        if socket.getdefaulttimeout() is None and super().gettimeout():
            sock.setblocking(True)
        return sock, addr

    @staticmethod
    def __pack_frame(frame):
        f = BytesIO()
        np.savez(f, frame=frame)

        packet_size = len(f.getvalue())
        header = '{0}:'.format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        return out


private_ip = "172.31.40.34"
host = socket.gethostbyaddr(private_ip)[0]
port = 12348
with NumpySocket() as s:
    s.bind((host, port))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"connected: {addr}")
        frame = conn.recv()
        print(time.time())
        print("array received")
        print(frame)

    print(f"disconnected: {addr}")
