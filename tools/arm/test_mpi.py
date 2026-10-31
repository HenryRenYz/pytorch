"""Launch with lx-run --mode mpi (native drun/PMIx), outside the source root."""

import json
import socket

import torch
import torch.distributed as dist


dist.init_process_group("mpi")
rank, size = dist.get_rank(), dist.get_world_size()
x = torch.tensor([rank + 1.0], dtype=torch.float64)
dist.all_reduce(x)
torch.testing.assert_close(x, torch.tensor([size * (size + 1) / 2], dtype=x.dtype))

parts = [torch.empty(1) for _ in range(size)]
dist.all_gather(parts, torch.tensor([float(rank)]))
torch.testing.assert_close(torch.cat(parts), torch.arange(size).float())

source = torch.arange(size * 8, dtype=torch.float32) + rank * 100
result = torch.empty_like(source)
dist.all_to_all_single(result, source)
expected = torch.cat([torch.arange(8).float() + sender * 100 + rank * 8 for sender in range(size)])
torch.testing.assert_close(result, expected)
dist.barrier()
print(json.dumps({"host": socket.gethostname(), "rank": rank, "world_size": size, "mpi": "PASS"}), flush=True)
dist.destroy_process_group()
