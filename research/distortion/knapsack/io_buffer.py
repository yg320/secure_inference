import os.path
import shutil
import torch


class IO_Buffer:
    def __init__(self, word_size, load=False, buffer_size=10, device=0, buffer_dir=None):
        self.buffer_size = buffer_size
        self.buffer_dir = buffer_dir
        if self.buffer_dir is None:
            self.buffer_dir = f"/tmp/buffer_dir/device_{device}"

        self.word_size = word_size
        self.buffer_init_value = 255
        self.cur_frame = 0
        self.dirty = False

        self.buffer_path_format = os.path.join(self.buffer_dir, "{}.pt")
        self.buffer = self.buffer_init_value * torch.ones(size=(self.buffer_size, self.word_size), dtype=torch.uint8)

        if not load:
            if os.path.exists(self.buffer_dir):
                shutil.rmtree(self.buffer_dir)

            os.makedirs(self.buffer_dir)
        else:
            self.reload(0)

    def get_channel_frame(self, channel):
        return channel // self.buffer_size

    def reload(self, channel_frame):
        if self.dirty:
            torch.save(f=self.buffer_path_format.format(self.cur_frame), obj=self.buffer)
            self.dirty = False

        if os.path.exists(self.buffer_path_format.format(channel_frame)):
            self.buffer[:] = torch.load(self.buffer_path_format.format(channel_frame))
        else:
            self.buffer[:] = self.buffer_init_value
        self.cur_frame = channel_frame

    def flush(self):
        self.reload(self.cur_frame)

    def __setitem__(self, channel, value):
        channel_frame = self.get_channel_frame(channel)
        if channel_frame != self.cur_frame:
            self.reload(channel_frame)
        self.buffer[channel % self.buffer_size] = value
        self.dirty = True

    def __getitem__(self, channel):
        channel_frame = self.get_channel_frame(channel)
        if channel_frame != self.cur_frame:
            self.reload(channel_frame)
        self.dirty = True  # The user can change the buffer later on
        return self.buffer[channel % self.buffer_size]
