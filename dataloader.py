from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Final

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt

from utils.reproducibility import fold_in_seed

_INT64_DTYPE: Final[np.dtype[np.int64]] = np.dtype(np.int64)


class TokenStreamDataloaderState(BaseModel):
    """
    Minimal, JSON-ish state for exact resumption of TokenStreamDataloader.

    The shard order is derived deterministically from (seed, epoch), so we only need:
    - epoch: which "pass" through the shards we're on
    - shard_pos: index into the per-epoch shard order
    - token_pos: offset inside the current shard's per-rank segment (after skip_tokens)
    """

    model_config = ConfigDict(extra="forbid")

    epoch: NonNegativeInt = 0
    shard_pos: NonNegativeInt = 0
    token_pos: NonNegativeInt = 0


class TokenStreamDataloaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    block_size: PositiveInt

    seed: int = 0
    shuffle: bool = False

    rank: NonNegativeInt = 0
    world_size: PositiveInt = 1

    skip_tokens: NonNegativeInt = 0
    bos_token_id: NonNegativeInt | None = None
    ignore_index: int = -1


def _segment_bounds(length: int, *, rank: int, world_size: int) -> tuple[int, int]:
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if world_size <= 0:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, world_size), got rank={rank} world_size={world_size}")

    base = length // world_size
    remainder = length % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    return start, end


class ExampleStreamDataloaderState(BaseModel):
    """
    Minimal, JSON-ish state for exact resumption of ExampleStreamDataloader.

    The per-epoch example order is derived deterministically from (seed, epoch), so we only need:
    - epoch: which "pass" through the examples we're on
    - example_pos: offset inside this rank's per-epoch example segment
    """

    model_config = ConfigDict(extra="forbid")

    epoch: NonNegativeInt = 0
    example_pos: NonNegativeInt = 0


class ExampleStreamDataloaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    block_size: PositiveInt

    seed: int = 0
    shuffle: bool = False

    rank: NonNegativeInt = 0
    world_size: PositiveInt = 1


def _affine_permutation_params(length: int, *, seed: int, epoch: int) -> tuple[int, int]:
    """
    Deterministically derive a simple bijection i -> (offset + stride*i) mod length.

    This avoids allocating/shuffling an explicit permutation vector (important for large example
    datasets) while still providing a full permutation when stride and length are coprime.
    """
    if length <= 0:
        raise ValueError(f"length must be >= 1, got {length}")
    if epoch < 0:
        raise ValueError(f"epoch must be >= 0, got {epoch}")
    if length == 1:
        return 0, 1

    offset = int(fold_in_seed(int(seed), epoch, 0) % length)
    stride = int(fold_in_seed(int(seed), epoch, 1) % length)
    if stride == 0:
        stride = 1
    while math.gcd(stride, length) != 1:
        stride += 1
        if stride >= length:
            stride = 1
    return offset, stride


class ExampleStreamDataloader:
    """
    Example-aligned streaming dataloader for 2D token arrays (num_examples x (block_size+1)).

    This is the sample-wise counterpart to TokenStreamDataloader: it never "cuts" an example
    across batch boundaries, which is important for datasets where each row is an independent
    supervised sample (e.g. arithmetics) and train loss is expected to reach ~0.
    """

    def __init__(
        self,
        *,
        ids: np.ndarray,
        config: ExampleStreamDataloaderConfig,
        device: torch.device | str,
        context_mask: np.ndarray | None = None,
        ignore_index: int = -1,
    ) -> None:
        if ids.ndim != 2:
            raise ValueError(f"ExampleStreamDataloader expects ids.ndim == 2, got {ids.ndim}")
        if int(ids.shape[0]) <= 0:
            raise ValueError(f"ExampleStreamDataloader requires at least 1 example, got shape={ids.shape!r}")
        seq_len = int(ids.shape[1])
        block_size = int(config.block_size)
        expected_seq_len = block_size + 1
        if seq_len != expected_seq_len:
            raise ValueError(f"Expected ids.shape[1] == block_size+1 ({expected_seq_len}), got {seq_len}")
        if context_mask is not None:
            if context_mask.shape != ids.shape:
                raise ValueError(f"context_mask.shape {context_mask.shape!r} must match ids.shape {ids.shape!r}")

        self._ids = ids
        self._context_mask = context_mask
        self._ignore_index = int(ignore_index)
        self._config = config
        self._device = torch.device(device)
        self._device_type = self._device.type
        self._use_cuda_optimizations = self._device_type == "cuda"

        self._num_examples = int(ids.shape[0])
        seg_start, seg_end = _segment_bounds(
            self._num_examples,
            rank=int(self._config.rank),
            world_size=int(self._config.world_size),
        )
        if seg_end <= seg_start:
            raise ValueError(
                "ExampleStreamDataloader received an empty per-rank example segment: "
                f"num_examples={self._num_examples} rank={int(self._config.rank)} "
                f"world_size={int(self._config.world_size)}"
            )
        self._seg_start = seg_start
        self._seg_end = seg_end

        self._epoch: int = 0
        self._example_pos: int = 0
        self._offset, self._stride = self._compute_permutation(self._epoch)
        self._validate_state()

    @property
    def config(self) -> ExampleStreamDataloaderConfig:
        return self._config

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "example_pos": self._example_pos}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        state = ExampleStreamDataloaderState.model_validate(state_dict)
        self._epoch = int(state.epoch)
        self._offset, self._stride = self._compute_permutation(self._epoch)
        self._example_pos = int(state.example_pos)
        self._validate_state()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(self._config.batch_size)
        block_size = int(self._config.block_size)
        seq_len = block_size + 1

        ids_batch, mask_batch = self._next_examples(batch_size)
        ids_t = torch.from_numpy(ids_batch)
        x_cpu = ids_t[:, :block_size]
        y_cpu = ids_t[:, 1:seq_len]

        if mask_batch is not None:
            mask_t = torch.from_numpy(mask_batch)
            mask_y = mask_t[:, 1:seq_len]
            y_cpu = y_cpu.masked_fill(mask_y, self._ignore_index)

        if self._use_cuda_optimizations:
            x = x_cpu.pin_memory().to(device=self._device, non_blocking=True)
            y = y_cpu.pin_memory().to(device=self._device, non_blocking=True)
        else:
            x = x_cpu.to(self._device)
            y = y_cpu.to(self._device)
        return x, y

    def _compute_permutation(self, epoch: int) -> tuple[int, int]:
        if not self._config.shuffle:
            return 0, 1
        return _affine_permutation_params(self._num_examples, seed=int(self._config.seed), epoch=epoch)

    def _next_examples(self, count: int) -> tuple[np.ndarray, np.ndarray | None]:
        if count <= 0:
            raise ValueError(f"count must be >= 1, got {count}")
        block_size = int(self._config.block_size)
        seq_len = block_size + 1

        ids_out = np.empty((count, seq_len), dtype=_INT64_DTYPE)
        mask_out: np.ndarray | None = None
        if self._context_mask is not None:
            mask_out = np.empty((count, seq_len), dtype=np.bool_)

        filled = 0
        while filled < count:
            seg_len = self._seg_end - self._seg_start
            remaining = seg_len - self._example_pos
            if remaining <= 0:
                self._advance_epoch()
                continue

            take = min(count - filled, remaining)
            global_pos = self._seg_start + np.arange(self._example_pos, self._example_pos + take, dtype=np.int64)

            if self._config.shuffle:
                row_ix = (self._offset + self._stride * global_pos) % self._num_examples
            else:
                row_ix = global_pos

            ids_view = np.asarray(self._ids[row_ix, :], dtype=np.int64)
            ids_out[filled : filled + take, :] = ids_view

            if mask_out is not None and self._context_mask is not None:
                mask_view = np.asarray(self._context_mask[row_ix, :], dtype=np.bool_)
                mask_out[filled : filled + take, :] = mask_view

            filled += take
            self._example_pos += take

        return ids_out, mask_out

    def _advance_epoch(self) -> None:
        self._epoch += 1
        self._example_pos = 0
        self._offset, self._stride = self._compute_permutation(self._epoch)

    def _validate_state(self) -> None:
        if self._epoch < 0:
            raise RuntimeError(f"epoch must be >= 0, got {self._epoch}")
        if self._example_pos < 0:
            raise RuntimeError(f"example_pos must be >= 0, got {self._example_pos}")
        seg_len = self._seg_end - self._seg_start
        if self._example_pos > seg_len:
            raise RuntimeError(f"example_pos must be <= {seg_len}, got {self._example_pos}")


class TokenStreamDataloader:
    """
    Nanochat-style token streaming dataloader for pre-tokenized 1D token arrays.

    Differences vs the legacy nanoGPT "random crop" get_batch:
    - Consumes the token stream sequentially (optionally shuffling shard order per epoch).
    - Supports exact resume via a small state_dict (epoch/shard_pos/token_pos).
    - Shards deterministically across DDP ranks by splitting each shard into contiguous segments.
    """

    def __init__(
        self,
        *,
        shards: Sequence[np.ndarray],
        config: TokenStreamDataloaderConfig,
        device: torch.device | str,
    ) -> None:
        if not shards:
            raise ValueError("TokenStreamDataloader requires at least 1 shard")
        self._shards: tuple[np.ndarray, ...] = tuple(shards)
        self._config = config
        self._device = torch.device(device)
        self._device_type = self._device.type
        self._use_cuda_optimizations = self._device_type == "cuda"

        self._epoch: int = 0
        self._shard_pos: int = 0
        self._token_pos: int = 0
        self._order: list[int] = self._compute_order(self._epoch)
        self._validate_state()

    @property
    def config(self) -> TokenStreamDataloaderConfig:
        return self._config

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "shard_pos": self._shard_pos, "token_pos": self._token_pos}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        state = TokenStreamDataloaderState.model_validate(state_dict)
        self._epoch = int(state.epoch)
        self._order = self._compute_order(self._epoch)
        self._shard_pos = int(state.shard_pos)
        self._token_pos = int(state.token_pos)
        self._validate_state()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(self._config.batch_size)
        block_size = int(self._config.block_size)
        needed_tokens = batch_size * block_size + 1

        tokens_out = np.empty((needed_tokens,), dtype=_INT64_DTYPE)
        filled = 0
        while filled < needed_tokens:
            shard = self._current_shard()
            shard_len = int(shard.shape[0])
            skip_tokens = int(self._config.skip_tokens)
            if skip_tokens > shard_len:
                raise ValueError(f"skip_tokens exceeds shard length: skip_tokens={skip_tokens} shard_len={shard_len}")

            seg_start, seg_end = _segment_bounds(
                shard_len - skip_tokens,
                rank=int(self._config.rank),
                world_size=int(self._config.world_size),
            )
            seg_start += skip_tokens
            seg_end += skip_tokens

            abs_pos = seg_start + self._token_pos
            remaining = seg_end - abs_pos
            if remaining <= 0:
                self._advance_shard()
                continue

            take = min(needed_tokens - filled, remaining)
            view = shard[abs_pos : abs_pos + take]
            tokens_out[filled : filled + take] = view
            filled += take
            self._token_pos += int(take)

        scratch = torch.from_numpy(tokens_out)
        if self._use_cuda_optimizations:
            scratch = scratch.pin_memory()
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        bos_token_id = self._config.bos_token_id
        if bos_token_id is not None:
            targets_cpu[targets_cpu == int(bos_token_id)] = int(self._config.ignore_index)
        inputs = inputs_cpu.view(batch_size, block_size).to(
            device=self._device, non_blocking=self._use_cuda_optimizations
        )
        targets = targets_cpu.view(batch_size, block_size).to(
            device=self._device, non_blocking=self._use_cuda_optimizations
        )
        return inputs, targets

    def _compute_order(self, epoch: int) -> list[int]:
        order = list(range(len(self._shards)))
        if not self._config.shuffle:
            return order
        seed = fold_in_seed(int(self._config.seed), epoch)
        rng = random.Random(seed)
        rng.shuffle(order)
        return order

    def _current_shard(self) -> np.ndarray:
        if not self._order:
            raise RuntimeError("TokenStreamDataloader has no shards")
        if self._shard_pos < 0 or self._shard_pos >= len(self._order):
            raise RuntimeError(f"Invalid shard_pos={self._shard_pos} for {len(self._order)} shards")
        return self._shards[self._order[self._shard_pos]]

    def _advance_shard(self) -> None:
        self._shard_pos += 1
        self._token_pos = 0
        if self._shard_pos >= len(self._order):
            self._epoch += 1
            self._order = self._compute_order(self._epoch)
            self._shard_pos = 0

    def _validate_state(self) -> None:
        if self._epoch < 0:
            raise RuntimeError(f"epoch must be >= 0, got {self._epoch}")
        if not self._order:
            raise RuntimeError("TokenStreamDataloader has no shard order")
        if self._shard_pos < 0 or self._shard_pos >= len(self._order):
            raise RuntimeError(f"shard_pos must be in [0, {len(self._order)}), got {self._shard_pos}")
        if self._token_pos < 0:
            raise RuntimeError(f"token_pos must be >= 0, got {self._token_pos}")


class FitOrCutTokenStreamDataloaderState(BaseModel):
    """
    Minimal, JSON-ish state for exact resumption of FitOrCutTokenStreamDataloader.

    The shard order is derived deterministically from (seed, epoch), so we only need:
    - epoch: which "pass" through the shards we're on
    - shard_pos: index into the per-epoch shard order
    - token_pos: offset inside the current shard's per-rank segment (after skip_tokens)
    - cursor: current packed-stream cursor (0..block_size-1)
    - doc_kept_end_pos: end offset (exclusive) of the current document's kept prefix (0 when idle)
    - doc_end_pos: end offset (exclusive) of the current document in the raw stream (0 when idle)
    """

    model_config = ConfigDict(extra="forbid")

    epoch: NonNegativeInt = 0
    shard_pos: NonNegativeInt = 0
    token_pos: NonNegativeInt = 0
    cursor: NonNegativeInt = 0
    doc_kept_end_pos: NonNegativeInt = 0
    doc_end_pos: NonNegativeInt = 0


class FitOrCutTokenStreamDataloaderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: PositiveInt
    block_size: PositiveInt
    fit_or_cut_threshold: PositiveInt

    doc_bos_token_id: NonNegativeInt
    doc_eos_token_id: NonNegativeInt

    seed: int = 0
    shuffle: bool = False

    rank: NonNegativeInt = 0
    world_size: PositiveInt = 1

    skip_tokens: NonNegativeInt = 0
    bos_token_id: NonNegativeInt | None = None
    ignore_index: int = -1


_SCAN_CHUNK_SIZE: Final[int] = 1_048_576


def _find_next_token_pos(
    tokens: np.ndarray,
    token_id: int,
    *,
    start: int,
    end: int,
    chunk_size: int = _SCAN_CHUNK_SIZE,
) -> int | None:
    if start >= end:
        return None
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    pos = start
    while pos < end:
        chunk_end = min(end, pos + chunk_size)
        matches = np.flatnonzero(tokens[pos:chunk_end] == token_id)
        if matches.size:
            return pos + int(matches[0])
        pos = chunk_end
    return None


def _find_last_token_pos(
    tokens: np.ndarray,
    token_id: int,
    *,
    start: int,
    end: int,
    chunk_size: int = _SCAN_CHUNK_SIZE,
) -> int | None:
    if start >= end:
        return None
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    pos = end
    while pos > start:
        chunk_start = max(start, pos - chunk_size)
        matches = np.flatnonzero(tokens[chunk_start:pos] == token_id)
        if matches.size:
            return chunk_start + int(matches[-1])
        pos = chunk_start
    return None


def _fit_or_cut_keep_len(*, cursor: int, doc_len: int, block_size: int, threshold: int) -> int:
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if threshold >= block_size:
        raise ValueError(f"threshold must be < block_size, got threshold={threshold} block_size={block_size}")
    if cursor < 0 or cursor >= block_size:
        raise ValueError(f"cursor must be in [0, block_size), got cursor={cursor} block_size={block_size}")
    if doc_len < 0:
        raise ValueError(f"doc_len must be >= 0, got {doc_len}")
    if doc_len == 0:
        return 0

    total = cursor + doc_len
    remainder = total % block_size
    if remainder == 0:
        return doc_len

    if total > block_size and remainder < threshold:
        keep_len = doc_len - remainder
        if keep_len <= 0:
            raise RuntimeError(
                "Fit-or-Cut truncation kept no tokens; this should be impossible when total > block_size "
                f"(cursor={cursor} doc_len={doc_len} block_size={block_size} remainder={remainder})."
            )
        return keep_len

    return doc_len


class FitOrCutTokenStreamDataloader:
    """
    Document-aware Fit-or-Cut token streaming dataloader for pre-tokenized 1D token arrays.

    This loader implements the packing logic described in `data/README.md` §2 at runtime:
    - Disk data stores a raw document stream: BOS ... EOS, repeated (no preprocessing-time packing).
    - The dataloader applies Fit-or-Cut on CPU to drop short wrap fragments and keep the training loop
      fully dense (no padding/masking required for packing).
    """

    def __init__(
        self,
        *,
        shards: Sequence[np.ndarray],
        config: FitOrCutTokenStreamDataloaderConfig,
        device: torch.device | str,
    ) -> None:
        if not shards:
            raise ValueError("FitOrCutTokenStreamDataloader requires at least 1 shard")
        self._shards: tuple[np.ndarray, ...] = tuple(shards)
        self._config = config
        self._device = torch.device(device)
        self._device_type = self._device.type
        self._use_cuda_optimizations = self._device_type == "cuda"

        self._epoch: int = 0
        self._shard_pos: int = 0
        self._token_pos: int = 0
        self._cursor: int = 0
        self._doc_kept_end_pos: int = 0
        self._doc_end_pos: int = 0
        self._order: list[int] = self._compute_order(self._epoch)

        self._cached_shard_pos: int | None = None
        self._cached_seg_start: int = 0
        self._cached_aligned_start_pos: int = 0
        self._cached_aligned_end_pos: int = 0

        self._validate_state()

    @property
    def config(self) -> FitOrCutTokenStreamDataloaderConfig:
        return self._config

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self._epoch,
            "shard_pos": self._shard_pos,
            "token_pos": self._token_pos,
            "cursor": self._cursor,
            "doc_kept_end_pos": self._doc_kept_end_pos,
            "doc_end_pos": self._doc_end_pos,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        state = FitOrCutTokenStreamDataloaderState.model_validate(state_dict)
        self._epoch = int(state.epoch)
        self._order = self._compute_order(self._epoch)
        self._shard_pos = int(state.shard_pos)
        self._token_pos = int(state.token_pos)
        self._cursor = int(state.cursor)
        self._doc_kept_end_pos = int(state.doc_kept_end_pos)
        self._doc_end_pos = int(state.doc_end_pos)
        self._cached_shard_pos = None
        self._validate_state()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(self._config.batch_size)
        block_size = int(self._config.block_size)
        needed_tokens = batch_size * block_size + 1

        tokens_out = np.empty((needed_tokens,), dtype=_INT64_DTYPE)
        filled = 0
        while filled < needed_tokens:
            shard = self._current_shard()
            seg_start, aligned_start_pos, aligned_end_pos = self._current_aligned_segment(shard)
            if aligned_end_pos <= aligned_start_pos:
                self._advance_shard()
                continue

            if self._token_pos < aligned_start_pos:
                self._token_pos = aligned_start_pos
            if self._token_pos >= aligned_end_pos:
                self._advance_shard()
                continue

            self._ensure_doc_bounds(shard, seg_start=seg_start, aligned_end_pos=aligned_end_pos)
            if self._doc_kept_end_pos == 0 or self._doc_end_pos == 0:
                continue
            if self._token_pos >= self._doc_kept_end_pos:
                self._token_pos = self._doc_end_pos
                self._doc_kept_end_pos = 0
                self._doc_end_pos = 0
                continue

            take = min(needed_tokens - filled, self._doc_kept_end_pos - self._token_pos)
            abs_pos = seg_start + self._token_pos
            view = shard[abs_pos : abs_pos + take]
            tokens_out[filled : filled + take] = view
            filled += int(take)
            self._token_pos += int(take)
            self._cursor = (self._cursor + int(take)) % block_size

        scratch = torch.from_numpy(tokens_out)
        if self._use_cuda_optimizations:
            scratch = scratch.pin_memory()
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        bos_token_id = self._config.bos_token_id
        if bos_token_id is not None:
            targets_cpu[targets_cpu == int(bos_token_id)] = int(self._config.ignore_index)
        inputs = inputs_cpu.view(batch_size, block_size).to(
            device=self._device, non_blocking=self._use_cuda_optimizations
        )
        targets = targets_cpu.view(batch_size, block_size).to(
            device=self._device, non_blocking=self._use_cuda_optimizations
        )
        return inputs, targets

    def _compute_order(self, epoch: int) -> list[int]:
        order = list(range(len(self._shards)))
        if not self._config.shuffle:
            return order
        seed = fold_in_seed(int(self._config.seed), epoch)
        rng = random.Random(seed)
        rng.shuffle(order)
        return order

    def _current_shard(self) -> np.ndarray:
        if not self._order:
            raise RuntimeError("FitOrCutTokenStreamDataloader has no shards")
        if self._shard_pos < 0 or self._shard_pos >= len(self._order):
            raise RuntimeError(f"Invalid shard_pos={self._shard_pos} for {len(self._order)} shards")
        return self._shards[self._order[self._shard_pos]]

    def _current_aligned_segment(self, shard: np.ndarray) -> tuple[int, int, int]:
        if self._cached_shard_pos == self._shard_pos:
            return self._cached_seg_start, self._cached_aligned_start_pos, self._cached_aligned_end_pos

        shard_len = int(shard.shape[0])
        skip_tokens = int(self._config.skip_tokens)
        if skip_tokens > shard_len:
            raise ValueError(f"skip_tokens exceeds shard length: skip_tokens={skip_tokens} shard_len={shard_len}")

        seg_start, seg_end = _segment_bounds(
            shard_len - skip_tokens,
            rank=int(self._config.rank),
            world_size=int(self._config.world_size),
        )
        seg_start += skip_tokens
        seg_end += skip_tokens

        bos = int(self._config.doc_bos_token_id)
        eos = int(self._config.doc_eos_token_id)
        aligned_start_abs = _find_next_token_pos(shard, bos, start=seg_start, end=seg_end)
        if aligned_start_abs is None:
            aligned_start_abs = seg_end
        aligned_end_abs = seg_end
        if aligned_start_abs < seg_end:
            last_eos = _find_last_token_pos(shard, eos, start=aligned_start_abs, end=seg_end)
            if last_eos is not None:
                aligned_end_abs = int(last_eos) + 1
            else:
                aligned_end_abs = aligned_start_abs

        self._cached_shard_pos = self._shard_pos
        self._cached_seg_start = seg_start
        self._cached_aligned_start_pos = aligned_start_abs - seg_start
        self._cached_aligned_end_pos = aligned_end_abs - seg_start
        return self._cached_seg_start, self._cached_aligned_start_pos, self._cached_aligned_end_pos

    def _ensure_doc_bounds(self, shard: np.ndarray, *, seg_start: int, aligned_end_pos: int) -> None:
        if self._doc_kept_end_pos != 0 and self._doc_end_pos != 0:
            return

        if self._token_pos >= aligned_end_pos:
            return

        bos = int(self._config.doc_bos_token_id)
        eos = int(self._config.doc_eos_token_id)
        if bos == eos:
            raise ValueError(f"doc_bos_token_id must differ from doc_eos_token_id, got {bos}.")

        abs_pos = seg_start + self._token_pos
        if int(shard[abs_pos]) != bos:
            next_bos = _find_next_token_pos(
                shard,
                bos,
                start=abs_pos + 1,
                end=seg_start + aligned_end_pos,
                chunk_size=65_536,
            )
            if next_bos is None:
                self._token_pos = aligned_end_pos
                return
            self._token_pos = int(next_bos) - seg_start
            abs_pos = int(next_bos)
            if self._token_pos >= aligned_end_pos:
                return

        next_eos = _find_next_token_pos(
            shard,
            eos,
            start=abs_pos + 1,
            end=seg_start + aligned_end_pos,
            chunk_size=65_536,
        )
        if next_eos is None:
            self._token_pos = aligned_end_pos
            return

        doc_end_abs = int(next_eos) + 1
        doc_len = doc_end_abs - abs_pos
        keep_len = _fit_or_cut_keep_len(
            cursor=int(self._cursor),
            doc_len=int(doc_len),
            block_size=int(self._config.block_size),
            threshold=int(self._config.fit_or_cut_threshold),
        )
        self._doc_kept_end_pos = self._token_pos + int(keep_len)
        self._doc_end_pos = self._token_pos + int(doc_len)

    def _advance_shard(self) -> None:
        self._shard_pos += 1
        self._token_pos = 0
        self._doc_kept_end_pos = 0
        self._doc_end_pos = 0
        self._cached_shard_pos = None
        if self._shard_pos >= len(self._order):
            self._epoch += 1
            self._order = self._compute_order(self._epoch)
            self._shard_pos = 0

    def _validate_state(self) -> None:
        block_size = int(self._config.block_size)
        threshold = int(self._config.fit_or_cut_threshold)
        if threshold >= block_size:
            raise ValueError(
                f"fit_or_cut_threshold must be < block_size, got threshold={threshold} block_size={block_size}"
            )
        if self._cursor < 0 or self._cursor >= block_size:
            raise RuntimeError(f"cursor must be in [0, {block_size}), got {self._cursor}")
        if self._epoch < 0:
            raise RuntimeError(f"epoch must be >= 0, got {self._epoch}")
        if not self._order:
            raise RuntimeError("FitOrCutTokenStreamDataloader has no shard order")
        if self._shard_pos < 0 or self._shard_pos >= len(self._order):
            raise RuntimeError(f"shard_pos must be in [0, {len(self._order)}), got {self._shard_pos}")
        if self._token_pos < 0:
            raise RuntimeError(f"token_pos must be >= 0, got {self._token_pos}")
        if (self._doc_kept_end_pos == 0) != (self._doc_end_pos == 0):
            raise RuntimeError(
                "doc_kept_end_pos and doc_end_pos must be both 0 (idle) or both non-zero (active), "
                f"got doc_kept_end_pos={self._doc_kept_end_pos} doc_end_pos={self._doc_end_pos}"
            )
        if self._doc_kept_end_pos != 0:
            if self._doc_kept_end_pos > self._doc_end_pos:
                raise RuntimeError(
                    f"doc_kept_end_pos must be <= doc_end_pos, got {self._doc_kept_end_pos} > {self._doc_end_pos}"
                )
