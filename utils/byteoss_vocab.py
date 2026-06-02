from __future__ import annotations

from typing import Final, Literal

import torch

from utils.nanogpt_tokenizers import BYTEOSS_SPECIAL_TOKEN_TO_ID

ByteossVocabMode = Literal["compact", "sparse"]

BYTEOSS_BYTE_VOCAB_SIZE: Final[int] = 256

BYTEOSS_SPECIAL_TOKEN_COUNT: Final[int] = len(BYTEOSS_SPECIAL_TOKEN_TO_ID)
BYTEOSS_SPARSE_SPECIAL_TOKEN_BASE_ID: Final[int] = min(BYTEOSS_SPECIAL_TOKEN_TO_ID.values())
BYTEOSS_SPARSE_SPECIAL_TOKEN_MAX_ID: Final[int] = max(BYTEOSS_SPECIAL_TOKEN_TO_ID.values())
BYTEOSS_SPARSE_VOCAB_SIZE: Final[int] = BYTEOSS_SPARSE_SPECIAL_TOKEN_MAX_ID + 1

if BYTEOSS_SPARSE_SPECIAL_TOKEN_MAX_ID - BYTEOSS_SPARSE_SPECIAL_TOKEN_BASE_ID + 1 != BYTEOSS_SPECIAL_TOKEN_COUNT:
    raise ValueError(
        "BYTEOSS special token ids must be contiguous to support compact remapping: "
        f"base={BYTEOSS_SPARSE_SPECIAL_TOKEN_BASE_ID} max={BYTEOSS_SPARSE_SPECIAL_TOKEN_MAX_ID} "
        f"count={BYTEOSS_SPECIAL_TOKEN_COUNT}."
    )

BYTEOSS_COMPACT_SPECIAL_TOKEN_OFFSET: Final[int] = BYTEOSS_BYTE_VOCAB_SIZE
BYTEOSS_COMPACT_SPECIAL_TOKEN_MAX_ID: Final[int] = (
    BYTEOSS_COMPACT_SPECIAL_TOKEN_OFFSET + BYTEOSS_SPECIAL_TOKEN_COUNT - 1
)
BYTEOSS_COMPACT_VOCAB_SIZE: Final[int] = BYTEOSS_COMPACT_SPECIAL_TOKEN_MAX_ID + 1


def byteoss_model_vocab_size(mode: ByteossVocabMode) -> int:
    return int(BYTEOSS_COMPACT_VOCAB_SIZE if mode == "compact" else BYTEOSS_SPARSE_VOCAB_SIZE)


def byteoss_compactify_ids_inplace(ids: torch.Tensor, *, validate: bool = False) -> None:
    """
    Remap ByteOSS token IDs to the compact ID space (0..BYTEOSS_COMPACT_VOCAB_SIZE-1) in-place.

    Supported input IDs:
    - Byte tokens: 0..255 (kept unchanged)
    - Compact special tokens: 256..(256+N-1) (kept unchanged)
    - Sparse special tokens: 199998..200018 (remapped to 256..)
    - Ignore index: -1 (kept unchanged)
    """
    if ids.numel() == 0:
        return

    sparse_special_base = int(BYTEOSS_SPARSE_SPECIAL_TOKEN_BASE_ID)
    sparse_special_limit = sparse_special_base + int(BYTEOSS_SPECIAL_TOKEN_COUNT)
    compact_special_base = int(BYTEOSS_COMPACT_SPECIAL_TOKEN_OFFSET)

    sparse_mask = (ids >= sparse_special_base) & (ids < sparse_special_limit)
    ids[sparse_mask] = ids[sparse_mask] - sparse_special_base + compact_special_base

    if not validate:
        return

    compact_special_limit = compact_special_base + int(BYTEOSS_SPECIAL_TOKEN_COUNT)
    allowed_mask = (
        (ids == -1)
        | ((ids >= 0) & (ids < int(BYTEOSS_BYTE_VOCAB_SIZE)))
        | ((ids >= compact_special_base) & (ids < compact_special_limit))
    )
    invalid_mask = ~allowed_mask
    if not bool(invalid_mask.any().item()):
        return

    bad = ids[invalid_mask]
    sample = bad[: min(10, int(bad.numel()))].tolist()
    raise ValueError(
        "Found invalid ByteOSS token id(s) while remapping to compact vocab. "
        f"Examples: {sample} (expected 0..{BYTEOSS_BYTE_VOCAB_SIZE - 1} for bytes, "
        f"{compact_special_base}..{compact_special_limit - 1} for special tokens, or -1 ignore index)."
    )
