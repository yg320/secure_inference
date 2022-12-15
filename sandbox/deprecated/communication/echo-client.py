# echo-client.py

import socket
import time
HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    time.sleep(1)
    s.sendall(b"0000000001")
    time.sleep(1)
    s.sendall(b"0000000001")
    time.sleep(1)
    s.sendall(b"0000000002")
    time.sleep(1)
    s.sendall(b"0000000003")
    time.sleep(1)
    s.sendall(b"0000000004")
    time.sleep(1)
    s.sendall(b"0000000005")
    time.sleep(1)
    s.sendall(b"0000000006")
    time.sleep(1)
    s.sendall(b"0000000007")
    time.sleep(1)
