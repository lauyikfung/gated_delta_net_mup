from __future__ import annotations

from typing import Final

import torch
import torch.distributed as dist

_INT64_DTYPE: Final[torch.dtype] = torch.int64
_UINT8_DTYPE: Final[torch.dtype] = torch.uint8


def broadcast_string(value: str, *, src: int = 0, device: torch.device | str = "cpu") -> str:
    """
    Broadcast a UTF-8 string from the `src` rank to all ranks in the current process group.

    This intentionally uses tensor broadcasts (not `broadcast_object_list`) so it works with
    both Gloo (CPU tensors) and NCCL (CUDA tensors).
    """
    if not dist.is_available() or not dist.is_initialized():
        return value
    if dist.get_world_size() == 1:
        return value

    comm_device = torch.device(device)
    rank = dist.get_rank()

    if rank == src:
        encoded = value.encode("utf-8")
        length = torch.tensor([len(encoded)], dtype=_INT64_DTYPE, device=comm_device)
    else:
        encoded = b""
        length = torch.tensor([0], dtype=_INT64_DTYPE, device=comm_device)

    dist.broadcast(length, src=src)
    num_bytes = int(length.item())

    if rank == src:
        payload = torch.tensor(list(encoded), dtype=_UINT8_DTYPE, device=comm_device)
    else:
        payload = torch.empty((num_bytes,), dtype=_UINT8_DTYPE, device=comm_device)

    dist.broadcast(payload, src=src)

    if rank == src:
        return value

    decoded = bytes(payload.to(device="cpu").tolist()).decode("utf-8")
    return decoded
