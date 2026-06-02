from __future__ import annotations

from typing import Final

import torch
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_IGNORE_INDEX: Final[int] = -1


class MaskedAccuracyCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    token_correct: int = Field(ge=0)
    token_total: int = Field(ge=0)
    sequence_correct: int = Field(ge=0)
    sequence_total: int = Field(ge=0)

    def token_accuracy(self) -> float:
        if self.token_total == 0:
            return 0.0
        return self.token_correct / self.token_total

    def sequence_accuracy(self) -> float:
        if self.sequence_total == 0:
            return 0.0
        return self.sequence_correct / self.sequence_total


def masked_accuracy_from_logits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> MaskedAccuracyCounts:
    """
    Compute token-level and exact-match (sequence-level) accuracy for teacher-forced LM logits.

    `targets` uses `ignore_index` to mark positions excluded from the accuracy computation (e.g. prompt tokens).
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected logits.ndim == 3 (B, T, V), got {logits.ndim}.")
    if targets.ndim != 2:
        raise ValueError(f"Expected targets.ndim == 2 (B, T), got {targets.ndim}.")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "Expected logits.shape[:2] == targets.shape, got "
            f"logits.shape={tuple(logits.shape)} targets.shape={tuple(targets.shape)}."
        )

    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index

    has_targets = mask.any(dim=1)
    per_example_all_correct = torch.logical_or(~mask, preds == targets).all(dim=1)

    counts = torch.stack(
        [
            ((preds == targets) & mask).sum(),
            mask.sum(),
            (per_example_all_correct & has_targets).sum(),
            has_targets.sum(),
        ]
    ).to(device="cpu")
    token_correct, token_total, sequence_correct, sequence_total = (int(value) for value in counts.tolist())

    return MaskedAccuracyCounts(
        token_correct=token_correct,
        token_total=token_total,
        sequence_correct=sequence_correct,
        sequence_total=sequence_total,
    )
