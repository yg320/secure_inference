import numpy as np
from tqdm import tqdm
import torch

from research.distortion.knapsack.io_buffer import IO_Buffer


class MultipleChoiceKnapsackSolver:
    @staticmethod
    def run_multiple_choice_knapsack(Ws, Ps, num_rows, num_columns, device=0, buffer_dir=None):
        Ws = torch.from_numpy(Ws)
        Ps = torch.from_numpy(Ps)

        assert num_columns < np.iinfo(np.int32).max
        arange = torch.arange(num_columns, dtype=torch.int64)
        indices = torch.zeros(size=(num_columns,), dtype=torch.int64)
        opt_buffer = - float("Inf") * torch.ones(size=(num_columns,), dtype=torch.float64)
        buffer = torch.zeros(size=(num_columns,), dtype=torch.float64)
        boolean_index_buffer = torch.zeros(size=(num_columns,), dtype=torch.bool)

        dp_arg = IO_Buffer(num_columns, buffer_size=1, device=device, buffer_dir=buffer_dir)
        dp = - float("Inf") * torch.ones(size=(num_columns + 1,), dtype=torch.float64)

        init_row = dp_arg[0].clone()
        init_row[Ws[0]] = torch.arange(Ws[0].shape[0], dtype=torch.uint8)
        dp_arg[0] = init_row
        dp[Ws[0]] = Ps[0]

        negative_one = -torch.ones(size=(1,), dtype=torch.int64)
        device = torch.device(f"cuda:{device}")

        Ws = Ws.to(device)  # (torch.Size([14272, 56]), torch.int64)                6.39M
        Ps = Ps.to(device)  # (torch.Size([14272, 56]), torch.float64)              6.39M
        arange = arange.to(device)  # (torch.Size([15656345, 1]), torch.int64)      125.25M
        indices = indices.to(device)  # (torch.Size([15656345, 56]), torch.int64)
        negative_one = negative_one.to(device)
        dp = dp.to(device)  # (torch.Size([15656346]), torch.float64)
        opt_buffer = opt_buffer.to(device)  # (torch.Size([876755320]), torch.float64)
        buffer = buffer.to(device)  # (torch.Size([876755320]), torch.float64)
        dp_arg.buffer = dp_arg.buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)
        boolean_index_buffer = boolean_index_buffer.to(device)  # (torch.Size([10, 15656345]), torch.uint8)

        for channel in tqdm(range(1, num_rows)):
            opt_buffer[:] = -float("Inf")
            for index in range(Ws.shape[1]):
                torch.sub(arange, Ws[channel][index], out=indices)
                torch.max(indices, negative_one, out=indices)

                torch.take(dp, indices, out=buffer)
                torch.add(buffer, Ps[channel][index], out=buffer)

                torch.gt(buffer, opt_buffer, out=boolean_index_buffer)
                dp_arg[channel][boolean_index_buffer] = index
                opt_buffer[boolean_index_buffer] = buffer[boolean_index_buffer]

            dp[:-1] = opt_buffer

        dp_arg.flush()
        return dp_arg, dp[:-1]

