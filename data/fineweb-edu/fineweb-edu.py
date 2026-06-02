"""
FineWeb-EDU dataset preprocessing (for SRS pretraining).
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import tiktoken
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.nanogpt_tokenizers import (
    DEFAULT_GPT_OSS_TOKENIZER_ID,
    TokenizerName,
    byteoss_encode,
    byteoss_encode_ordinary,
    gpt_oss_encode,
    gpt_oss_encode_ordinary,
    numpy_token_dtype,
    tokenizer_meta,
)

MAGIC = 20240520
DATAFILE_VERSION = 1
HEADER_INTS = 256
MAX_UINT16 = 2**16
HEADER_BYTES = HEADER_INTS * 4

FinewebEduVersion = Literal["10B", "100B", "350B"]
SUPPORTED_VERSIONS: tuple[FinewebEduVersion, ...] = ("10B", "100B", "350B")

_PRESETS: dict[FinewebEduVersion, tuple[str, str]] = {
    "10B": ("fineweb-edu10B", "sample-10BT"),
    "100B": ("fineweb-edu100B", "sample-100BT"),
    "350B": ("fineweb-edu350B", "sample-350BT"),
}

_SCRIPT_DIR = Path(__file__).resolve().parent

_ENC: tiktoken.Encoding | None = None
_EOT: int | None = None
_TOKENIZER_META: dict[str, object] | None = None
_TOKEN_DTYPE: np.dtype | None = None
_ALLOW_SPECIAL_TOKENS: bool = False


class FinewebEduPreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: FinewebEduVersion = "100B"
    shard_size: PositiveInt = Field(default=10**8, description="Size of each shard in tokens.")
    num_procs: PositiveInt | None = Field(default=None, description="Multiprocessing workers (default: auto).")
    chunksize: PositiveInt = Field(default=16, description="Multiprocessing Pool.imap() chunksize.")
    val_shards: PositiveInt = Field(default=1, description="Number of initial shards reserved for validation.")
    streaming: bool = Field(default=False, description="Use Hugging Face streaming mode (no local download).")
    hf_cache_dir: Path | None = Field(default=None, description="Hugging Face datasets cache dir (optional).")
    output_dir: Path | None = Field(default=None, description="Output directory (defaults to version preset).")
    tokenizer: TokenizerName = Field(default="gpt2", description="Tokenizer: gpt2|gpt-oss|byteoss.")
    encoding: str = Field(default="gpt2", description="tiktoken encoding name.")
    hf_tokenizer_id: str = Field(
        default=DEFAULT_GPT_OSS_TOKENIZER_ID,
        description="Hugging Face tokenizer ID (used when tokenizer='gpt-oss').",
    )
    allow_special_tokens: bool = Field(
        default=False,
        description="Allow parsing special token strings inside document text (default: False; treat as literal).",
    )
    max_docs: PositiveInt | None = Field(default=None, description="Stop after N documents (debugging).")

    # Aligned streaming/packing ("Fit or Cut") during `.bin` writing.
    fit_or_cut: bool = Field(
        default=False,
        description="Enable Fit-or-Cut smart truncation to avoid short orphan fragments at the start of blocks.",
    )
    block_size: PositiveInt = Field(
        default=4096,
        description="Model context window used for Fit-or-Cut packing (must match training `block_size`).",
    )
    fit_or_cut_threshold: PositiveInt = Field(
        default=100,
        description="Minimum viable fragment length (remainders smaller than this are truncated on wrap).",
    )

    @model_validator(mode="after")
    def _validate_fit_or_cut(self) -> FinewebEduPreprocessConfig:
        if not self.fit_or_cut:
            return self
        block_size = int(self.block_size)
        threshold = int(self.fit_or_cut_threshold)
        if threshold >= block_size:
            raise ValueError(f"fit_or_cut_threshold must be < block_size, got threshold={threshold} block_size={block_size}")
        return self

    def preset(self) -> tuple[str, str]:
        return _PRESETS[self.version]

    def resolved_output_dir(self) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        local_dir, _ = self.preset()
        return _SCRIPT_DIR / local_dir


def _default_num_procs() -> int:
    cpu_count = os.cpu_count() or 1
    return min(32, max(1, cpu_count - 2))


def _init_tokenizer(
    tokenizer: TokenizerName,
    encoding_name: str,
    hf_tokenizer_id: str = DEFAULT_GPT_OSS_TOKENIZER_ID,
    allow_special_tokens: bool = False,
) -> None:
    global _ALLOW_SPECIAL_TOKENS, _ENC, _EOT, _TOKENIZER_META, _TOKEN_DTYPE
    _ALLOW_SPECIAL_TOKENS = bool(allow_special_tokens)
    meta = tokenizer_meta(tokenizer, encoding=encoding_name, hf_tokenizer_id=hf_tokenizer_id)
    _TOKENIZER_META = meta.model_dump()
    _TOKEN_DTYPE = numpy_token_dtype(meta.token_dtype)

    if tokenizer == "gpt2":
        _ENC = tiktoken.get_encoding(encoding_name)
        _EOT = _ENC.eot_token
        bos_id = int(meta.bos_token_id)
        eos_id = int(meta.eos_token_id)

        max_token_value = getattr(_ENC, "max_token_value", None)
        if max_token_value is None:
            max_token_value = _ENC.n_vocab - 1
        if int(max_token_value) >= MAX_UINT16:
            raise ValueError(
                f"Encoding {encoding_name!r} produces tokens up to {max_token_value}, "
                f"but this script writes uint16 (< {MAX_UINT16})."
            )
        if _EOT is None or _EOT < 0 or _EOT >= MAX_UINT16:
            raise ValueError(f"Invalid eot_token={_EOT} for uint16 output.")
        if bos_id < 0 or bos_id >= MAX_UINT16:
            raise ValueError(f"Invalid bos_token_id={bos_id} for uint16 output.")
        if eos_id < 0 or eos_id >= MAX_UINT16:
            raise ValueError(f"Invalid eos_token_id={eos_id} for uint16 output.")
        if bos_id == eos_id:
            raise ValueError(f"bos_token_id must differ from eos_token_id, got {bos_id}.")
        return

    if tokenizer == "byteoss":
        _ENC = None
        _EOT = None
        return

    if tokenizer == "gpt-oss":
        _ENC = None
        _EOT = None
        return

    raise ValueError(f"Unsupported tokenizer={tokenizer!r}")


def tokenize(doc: Mapping[str, Any]) -> np.ndarray:
    """Tokenize a single document; returns a 1D numpy array of token ids (BOS...EOS wrapped)."""
    if _TOKENIZER_META is None or _TOKEN_DTYPE is None:
        raise RuntimeError("Tokenizer not initialized. This should run via mp.Pool(..., initializer=_init_tokenizer).")

    text = doc.get("text", "")
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    tokenizer_name = str(_TOKENIZER_META["tokenizer"])
    if tokenizer_name == "gpt2":
        if _ENC is None:
            raise RuntimeError("Expected tiktoken encoder initialized for tokenizer='gpt2'.")
        bos_id = int(_TOKENIZER_META["bos_token_id"])
        eos_id = int(_TOKENIZER_META["eos_token_id"])
        if _ALLOW_SPECIAL_TOKENS:
            bos_token = str(_TOKENIZER_META["bos_token"])
            parts = text.split(bos_token)
            ids: list[int] = []
            ids.extend(_ENC.encode(parts[0], allowed_special="all"))
            for part in parts[1:]:
                ids.append(bos_id)
                ids.extend(_ENC.encode(part, allowed_special="all"))
        else:
            ids = _ENC.encode_ordinary(text)
        out = np.empty((len(ids) + 2,), dtype=_TOKEN_DTYPE)
        out[0] = bos_id
        out[1:-1] = np.asarray(ids, dtype=_TOKEN_DTYPE)
        out[-1] = eos_id
        return out

    if tokenizer_name == "byteoss":
        bos_id = int(_TOKENIZER_META["bos_token_id"])
        eos_id = int(_TOKENIZER_META["eos_token_id"])
        ids = [bos_id]
        if _ALLOW_SPECIAL_TOKENS:
            ids.extend(byteoss_encode(text))
        else:
            ids.extend(byteoss_encode_ordinary(text))
        ids.append(eos_id)
        return np.asarray(ids, dtype=_TOKEN_DTYPE)

    if tokenizer_name == "gpt-oss":
        bos_id = int(_TOKENIZER_META["bos_token_id"])
        eos_id = int(_TOKENIZER_META["eos_token_id"])
        hf_tokenizer_id = str(_TOKENIZER_META.get("hf_tokenizer_id") or DEFAULT_GPT_OSS_TOKENIZER_ID)
        ids = [bos_id]
        if _ALLOW_SPECIAL_TOKENS:
            ids.extend(gpt_oss_encode(text, hf_tokenizer_id=hf_tokenizer_id))
        else:
            ids.extend(gpt_oss_encode_ordinary(text, hf_tokenizer_id=hf_tokenizer_id))
        ids.append(eos_id)
        return np.asarray(ids, dtype=_TOKEN_DTYPE)

    raise ValueError(f"Unsupported tokenizer={tokenizer_name!r}")


def _build_header(token_count: int) -> np.ndarray:
    if token_count >= 2**31:
        raise ValueError(f"token count too large: {token_count} (must be < 2**31)")
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = token_count
    return header


def write_datafile(path: Path, tokens: np.ndarray, *, token_dtype: np.dtype) -> None:
    """
    Saves token data as a `.bin` file, for reading in C / NumPy memmaps.

    Format:
    - 256 int32 header values (1024 bytes)
    - token stream as uint16 / uint32 (configurable)
    """
    if tokens.ndim != 1:
        raise ValueError(f"expected 1D tokens array, got shape={tokens.shape}")
    if tokens.dtype != token_dtype:
        tokens = tokens.astype(token_dtype, copy=False)

    token_count = int(tokens.size)
    header = _build_header(token_count)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.tmp")
    print(f"writing {token_count:,} tokens to {path}")
    with open(tmp_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    os.replace(tmp_path, path)


def _shard_path(output_dir: Path, split: str, shard_index: int) -> Path:
    return output_dir / f"fineweb_{split}_{shard_index:06d}.bin"


def _fit_or_cut_decision(*, cursor: int, doc_len: int, block_size: int, threshold: int) -> tuple[int, int]:
    """
    Decide how many tokens to keep for Fit-or-Cut packing.

    The packing cursor is the current fill level of the active block (0..block_size-1).
    If writing `doc_len` tokens would wrap into a new block and leave a short tail fragment
    at the start of that new block (0 < remainder < threshold), we drop the last `remainder`
    tokens so the document ends exactly at a block boundary.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if cursor < 0 or cursor >= block_size:
        raise ValueError(f"cursor must be in [0, block_size), got cursor={cursor} block_size={block_size}")
    if doc_len < 0:
        raise ValueError(f"doc_len must be >= 0, got {doc_len}")
    if doc_len == 0:
        return 0, cursor

    total = cursor + doc_len
    remainder = total % block_size

    if remainder == 0:
        return doc_len, 0

    if total > block_size and remainder < threshold:
        kept_len = doc_len - remainder
        if kept_len <= 0:
            raise RuntimeError(
                "Fit-or-Cut truncation kept no tokens; this should be impossible when total > block_size "
                f"(cursor={cursor} doc_len={doc_len} block_size={block_size} remainder={remainder})."
            )
        return kept_len, 0

    return doc_len, remainder


def _write_shards(
    token_stream: Iterable[np.ndarray],
    *,
    output_dir: Path,
    shard_size: int,
    val_shards: int,
    token_dtype: np.dtype,
    fit_or_cut: bool,
    block_size: int,
    fit_or_cut_threshold: int,
) -> None:
    try:
        shard_buffer = np.empty((shard_size,), dtype=token_dtype)
    except MemoryError as exc:
        shard_gib = shard_size * int(np.dtype(token_dtype).itemsize) / (1024**3)
        raise MemoryError(
            f"Failed to allocate shard buffer: shard_size={shard_size} {token_dtype} tokens (~{shard_gib:.2f} GiB). "
            "Reduce --shard_size."
        ) from exc

    shard_index = 0
    tokens_in_shard = 0
    progress_bar: tqdm | None = None
    cursor = 0
    truncated_docs = 0
    truncated_tokens = 0
    block_size_i = int(block_size)
    threshold_i = int(fit_or_cut_threshold)

    def _flush_current_shard(*, token_count: int) -> None:
        nonlocal shard_index, tokens_in_shard, progress_bar
        if token_count <= 0:
            return
        split = "val" if shard_index < val_shards else "train"
        path = _shard_path(output_dir, split, shard_index)
        if progress_bar is not None:
            progress_bar.close()
            progress_bar = None
        write_datafile(path, shard_buffer[:token_count], token_dtype=token_dtype)
        shard_index += 1
        tokens_in_shard = 0

    for tokens in token_stream:
        doc_len = int(tokens.size)
        if doc_len <= 0:
            continue

        if fit_or_cut:
            kept_len, cursor = _fit_or_cut_decision(
                cursor=cursor,
                doc_len=doc_len,
                block_size=block_size_i,
                threshold=threshold_i,
            )
            if kept_len <= 0:
                continue
            if kept_len != doc_len:
                truncated_docs += 1
                truncated_tokens += doc_len - kept_len
                tokens = tokens[:kept_len]
                doc_len = kept_len

        # Ensure that shard boundaries never cut a document. If the document would overflow the
        # current shard, seal it early (even if it's underfilled) and start a new shard.
        if doc_len > shard_size:
            if tokens_in_shard > 0:
                _flush_current_shard(token_count=tokens_in_shard)
            split = "val" if shard_index < val_shards else "train"
            path = _shard_path(output_dir, split, shard_index)
            write_datafile(path, tokens, token_dtype=token_dtype)
            shard_index += 1
            continue

        if tokens_in_shard > 0 and tokens_in_shard + doc_len > shard_size:
            _flush_current_shard(token_count=tokens_in_shard)

        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

        shard_buffer[tokens_in_shard : tokens_in_shard + doc_len] = tokens
        tokens_in_shard += doc_len
        progress_bar.update(doc_len)

        if tokens_in_shard == shard_size:
            _flush_current_shard(token_count=shard_size)

    if tokens_in_shard > 0:
        _flush_current_shard(token_count=tokens_in_shard)
    elif progress_bar is not None:
        progress_bar.close()

    if fit_or_cut and truncated_docs > 0:
        print(f"fit-or-cut truncated {truncated_docs:,} docs / {truncated_tokens:,} tokens")


def _parse_config(argv: Sequence[str] | None = None) -> FinewebEduPreprocessConfig:
    parser = argparse.ArgumentParser(description="FineWeb-EDU dataset preprocessing")
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="100B",
        choices=SUPPORTED_VERSIONS,
        help="Which version to use: 10B|100B|350B",
    )
    parser.add_argument("-s", "--shard_size", type=int, default=10**8, help="Size of each shard in tokens")
    parser.add_argument("--num_procs", type=int, default=None, help="Multiprocessing workers (default: auto)")
    parser.add_argument("--chunksize", type=int, default=16, help="Multiprocessing Pool.imap() chunksize")
    parser.add_argument("--val_shards", type=int, default=1, help="Number of initial shards reserved for validation")
    parser.add_argument("--streaming", action="store_true", help="Stream from HF instead of downloading")
    parser.add_argument("--hf_cache_dir", type=Path, default=None, help="HF datasets cache directory (optional)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory (optional)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["gpt2", "gpt-oss", "byteoss"], help="Tokenizer name")
    parser.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name")
    parser.add_argument(
        "--hf_tokenizer_id",
        type=str,
        default=DEFAULT_GPT_OSS_TOKENIZER_ID,
        help=f"Hugging Face tokenizer ID for tokenizer='gpt-oss' (default: {DEFAULT_GPT_OSS_TOKENIZER_ID}).",
    )
    parser.add_argument(
        "--allow_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow parsing special token strings inside document text (default: False).",
    )
    parser.add_argument("--max_docs", type=int, default=None, help="Stop after N documents (debugging)")
    parser.add_argument(
        "--fit_or_cut",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Fit-or-Cut packing (default: False). Use --fit_or_cut to enable.",
    )
    parser.add_argument("--block_size", type=int, default=4096, help="Model context window for Fit-or-Cut packing.")
    parser.add_argument("--fit_or_cut_threshold", type=int, default=100, help="Minimum viable fragment length.")

    ns = parser.parse_args(argv)
    try:
        return FinewebEduPreprocessConfig.model_validate(vars(ns))
    except ValidationError as exc:
        parser.error(str(exc))
        raise  # unreachable, but keeps type checkers happy


def main() -> None:
    config = _parse_config()
    output_dir = config.resolved_output_dir()
    local_dir, remote_name = config.preset()
    nprocs = int(config.num_procs or _default_num_procs())
    meta = tokenizer_meta(config.tokenizer, encoding=config.encoding, hf_tokenizer_id=config.hf_tokenizer_id)
    token_dtype = numpy_token_dtype(meta.token_dtype)

    print(
        "FineWeb-EDU preprocessing config:\n"
        f"- version: {config.version} (HF name={remote_name}, local_dir={local_dir})\n"
        f"- output_dir: {output_dir}\n"
        f"- shard_size: {config.shard_size:,} tokens\n"
        f"- val_shards: {config.val_shards}\n"
        f"- streaming: {config.streaming}\n"
        f"- num_procs: {nprocs}\n"
        f"- chunksize: {config.chunksize}\n"
        f"- tokenizer: {config.tokenizer}\n"
        f"- encoding: {config.encoding!r}\n"
        f"- hf_tokenizer_id: {config.hf_tokenizer_id!r}\n"
        f"- allow_special_tokens: {config.allow_special_tokens}\n"
        f"- max_docs: {config.max_docs}\n"
        f"- fit_or_cut: {config.fit_or_cut}\n"
        f"- block_size: {config.block_size}\n"
        f"- fit_or_cut_threshold: {config.fit_or_cut_threshold}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.pkl"
    payload = meta.model_dump() | {
        "tokenization": {"allow_special_tokens": bool(config.allow_special_tokens)},
        "datafile": {
            "magic": MAGIC,
            "version": DATAFILE_VERSION,
            "header_ints": HEADER_INTS,
            "header_bytes": HEADER_BYTES,
        },
        "packing": {
            "algorithm": "fit_or_cut",
            "enabled": bool(config.fit_or_cut),
            "block_size": int(config.block_size),
            "threshold": int(config.fit_or_cut_threshold),
        },
    }
    meta_tmp = Path(f"{meta_path}.tmp")
    meta_tmp.write_bytes(pickle.dumps(payload))
    os.replace(meta_tmp, meta_path)

    load_kwargs: dict[str, Any] = {"name": remote_name, "split": "train", "streaming": config.streaming}
    if config.hf_cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.hf_cache_dir)
    fw = load_dataset("HuggingFaceFW/fineweb-edu", **load_kwargs)

    docs_iter = fw
    if config.max_docs is not None:
        docs_iter = islice(docs_iter, int(config.max_docs))

    with mp.Pool(
        nprocs,
        initializer=_init_tokenizer,
        initargs=(config.tokenizer, config.encoding, config.hf_tokenizer_id, bool(config.allow_special_tokens)),
    ) as pool:
        token_stream = pool.imap(tokenize, docs_iter, chunksize=int(config.chunksize))
        _write_shards(
            token_stream,
            output_dir=output_dir,
            shard_size=int(config.shard_size),
            val_shards=int(config.val_shards),
            token_dtype=token_dtype,
            fit_or_cut=bool(config.fit_or_cut),
            block_size=int(config.block_size),
            fit_or_cut_threshold=int(config.fit_or_cut_threshold),
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()
