from threading import Thread
import queue

class PRF(Thread):
    def __init__(self):
        super(PRF, self).__init__()
        self.numpy_arr_queue = queue.Queue()

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
        t0 = time.time()
        arr = self.numpy_arr_queue.get()
        t1 = time.time()
        self.get_total_time += (t1-t0)
        return arr


if __name__ == "__main__":
    mrng = MultithreadedRNG(10000000000, threads=1, seed=12345)
    mrng.fill()
