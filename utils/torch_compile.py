from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch

_LogFn = Callable[[str], None]


def maybe_torch_compile(
    model: torch.nn.Module,
    *,
    enabled: bool,
    log: _LogFn | None = print,
) -> torch.nn.Module:
    """
    Best-effort wrapper around `torch.compile`.

    `torch.compile` is an optional optimization. Some environments (for example, Python 3.14+)
    raise at runtime when compilation is requested. In those cases we continue training with the
    uncompiled model.
    """
    if not enabled:
        return model

    if log is not None:
        log("compiling the model... (takes a ~minute)")

    try:
        compiled = torch.compile(model)
        return cast(torch.nn.Module, compiled)
    except RuntimeError as exc:
        message = str(exc)
        if "torch.compile is not supported" in message:
            if log is not None:
                log(f"torch.compile unavailable: {message}; continuing without compilation.")
            return model
        raise
