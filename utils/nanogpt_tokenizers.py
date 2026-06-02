"""Tokenizer metadata and helpers for preprocessing and training."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
import tiktoken
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizerBackend
    from transformers import PreTrainedTokenizerFast

TokenizerName = Literal["gpt2", "byteoss", "gpt-oss"]
TokenDTypeName = Literal["uint16", "uint32"]

GPT2_BOS_TOKEN: Final[str] = "<|beginoftext|>"
GPT2_EOS_TOKEN: Final[str] = "<|endoftext|>"

DEFAULT_GPT_OSS_TOKENIZER_ID: Final[str] = "openai/gpt-oss-120b"


BYTEOSS_SPECIAL_TOKEN_TO_ID: Final[dict[str, int]] = {
    "<|startoftext|>": 199998,
    "<|endoftext|>": 199999,
    "<|reserved_200000|>": 200000,
    "<|reserved_200001|>": 200001,
    "<|return|>": 200002,
    "<|constrain|>": 200003,
    "<|reserved_200004|>": 200004,
    "<|channel|>": 200005,
    "<|start|>": 200006,
    "<|end|>": 200007,
    "<|message|>": 200008,
    "<|reserved_200009|>": 200009,
    "<|reserved_200010|>": 200010,
    "<|reserved_200011|>": 200011,
    "<|call|>": 200012,
    "<|reserved_200013|>": 200013,
    "<|reserved_200014|>": 200014,
    "<|reserved_200015|>": 200015,
    "<|reserved_200016|>": 200016,
    "<|reserved_200017|>": 200017,
    "<|endofprompt|>": 200018,
}

BYTEOSS_BOS_TOKEN: Final[str] = "<|startoftext|>"
BYTEOSS_EOS_TOKEN: Final[str] = "<|return|>"
BYTEOSS_PAD_TOKEN: Final[str] = "<|endoftext|>"

BYTEOSS_BOS_TOKEN_ID: Final[int] = BYTEOSS_SPECIAL_TOKEN_TO_ID[BYTEOSS_BOS_TOKEN]
BYTEOSS_EOS_TOKEN_ID: Final[int] = BYTEOSS_SPECIAL_TOKEN_TO_ID[BYTEOSS_EOS_TOKEN]
BYTEOSS_PAD_TOKEN_ID: Final[int] = BYTEOSS_SPECIAL_TOKEN_TO_ID[BYTEOSS_PAD_TOKEN]
BYTEOSS_VOCAB_SIZE: Final[int] = max(BYTEOSS_SPECIAL_TOKEN_TO_ID.values()) + 1

_BYTEOSS_SPECIAL_TOKENS_SORTED: Final[tuple[str, ...]] = tuple(
    sorted(BYTEOSS_SPECIAL_TOKEN_TO_ID.keys(), key=len, reverse=True)
)


class TokenizerMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokenizer: TokenizerName
    vocab_size: int
    token_dtype: TokenDTypeName

    encoding: str | None = None
    hf_tokenizer_id: str | None = None

    bos_token: str
    bos_token_id: int
    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int

    special_tokens: dict[str, int] | None = None


def byteoss_encode(text: str) -> list[int]:
    """
    UTF-8 byte tokenizer (0..255) with GPT-OSS special tokens.

    Any known `<|...|>` special token substring is emitted as a single token id; all
    other text is encoded to UTF-8 bytes and emitted as 0..255 token ids.
    """
    ids: list[int] = []
    cursor = 0

    while cursor < len(text):
        next_special = text.find("<|", cursor)
        if next_special == -1:
            ids.extend(text[cursor:].encode("utf-8"))
            break

        if next_special > cursor:
            ids.extend(text[cursor:next_special].encode("utf-8"))

        matched = False
        for token in _BYTEOSS_SPECIAL_TOKENS_SORTED:
            if text.startswith(token, next_special):
                ids.append(BYTEOSS_SPECIAL_TOKEN_TO_ID[token])
                cursor = next_special + len(token)
                matched = True
                break

        if not matched:
            ids.extend("<|".encode("utf-8"))
            cursor = next_special + 2

    return ids


def byteoss_encode_ordinary(text: str) -> list[int]:
    """
    Encode text as raw UTF-8 bytes (0..255) without interpreting any `<|...|>` special token strings.

    This is useful for web-style corpora (OpenWebText/FineWeb-EDU) where raw documents may contain
    substrings like `<|startoftext|>`/`<|return|>`/`<|endoftext|>` that must be treated as literal text
    and not inject special token IDs into the token stream.
    """
    return list(text.encode("utf-8"))


@lru_cache(maxsize=None)
def _get_hf_tokenizer(hf_tokenizer_id: str) -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_id, use_fast=True)
    if tokenizer is None:
        raise RuntimeError(f"Failed to load HF tokenizer: {hf_tokenizer_id!r}")
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise TypeError(
            f"Expected a fast HF tokenizer for {hf_tokenizer_id!r}, got {type(tokenizer).__name__}. "
            "Install `tokenizers` and retry."
        )
    return tokenizer


def _hf_tokenizer_max_id(tokenizer: PreTrainedTokenizerFast) -> int:
    base_vocab_size = int(getattr(tokenizer, "vocab_size", 0))
    base_max = base_vocab_size - 1

    added_vocab_raw = tokenizer.get_added_vocab()
    added_max = max((int(i) for i in added_vocab_raw.values()), default=-1)

    special_max = max((int(i) for i in tokenizer.all_special_ids), default=-1)
    max_id = max(base_max, added_max, special_max)
    if max_id < 0:
        raise ValueError("Unable to determine tokenizer max id; got no vocab or added/special tokens.")
    return max_id


@lru_cache(maxsize=None)
def _get_hf_backend_tokenizer_without_added_tokens(hf_tokenizer_id: str) -> HFTokenizerBackend:
    """
    Return a `tokenizers.Tokenizer` backend copy with *all* added tokens removed.

    This is the HF equivalent of `tiktoken.encode_ordinary`: it prevents substrings like
    `<|startoftext|>` from being treated as control tokens inside raw corpus text, while
    still allowing the model to generate them as literal strings.
    """
    import json

    from tokenizers import Tokenizer

    hf_tokenizer = _get_hf_tokenizer(hf_tokenizer_id)
    backend = hf_tokenizer.backend_tokenizer
    payload = json.loads(backend.to_str())
    payload["added_tokens"] = []
    return Tokenizer.from_str(json.dumps(payload))


def gpt_oss_encode_ordinary(text: str, *, hf_tokenizer_id: str = DEFAULT_GPT_OSS_TOKENIZER_ID) -> list[int]:
    """Encode text using the GPT-OSS HF tokenizer without interpreting any added/special tokens."""
    backend = _get_hf_backend_tokenizer_without_added_tokens(hf_tokenizer_id)
    encoding = backend.encode(text, add_special_tokens=False)
    return list(encoding.ids)


def gpt_oss_encode(text: str, *, hf_tokenizer_id: str = DEFAULT_GPT_OSS_TOKENIZER_ID) -> list[int]:
    """
    Encode text using the GPT-OSS HF tokenizer, including any added/special tokens present in the text.

    Note: callers that manually wrap BOS/EOS should pass `add_special_tokens=False` (as done here) to avoid
    double-inserting boundary tokens.
    """
    hf_tokenizer = _get_hf_tokenizer(hf_tokenizer_id)
    ids = hf_tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in ids]


def gpt2_encode(text: str, *, encoding: str = "gpt2", allow_special_tokens: bool = False) -> list[int]:
    """
    Encode text using the `tiktoken` GPT-2 BPE.

    - When `allow_special_tokens=False` (default), special token strings like `<|endoftext|>` are treated as
      ordinary text and do not inject special token IDs.
    - When `allow_special_tokens=True`, special tokens are parsed, and this repo's GPT-2 BOS marker
      `<|beginoftext|>` is additionally recognized and emitted as a distinct BOS token id.
    """
    enc = _get_tiktoken_encoding(encoding)
    if not allow_special_tokens:
        return list(enc.encode_ordinary(text))

    bos_token = GPT2_BOS_TOKEN
    bos_id = int(getattr(enc, "n_vocab"))

    parts = text.split(bos_token)
    ids: list[int] = []
    ids.extend(enc.encode(parts[0], allowed_special="all"))
    for part in parts[1:]:
        ids.append(bos_id)
        ids.extend(enc.encode(part, allowed_special="all"))
    return [int(token_id) for token_id in ids]


def gpt_oss_meta(*, hf_tokenizer_id: str = DEFAULT_GPT_OSS_TOKENIZER_ID) -> TokenizerMeta:
    hf_tokenizer = _get_hf_tokenizer(hf_tokenizer_id)
    bos_token = hf_tokenizer.bos_token
    eos_token = hf_tokenizer.eos_token
    pad_token = hf_tokenizer.pad_token

    if not isinstance(bos_token, str) or not bos_token:
        raise ValueError(f"HF tokenizer {hf_tokenizer_id!r} is missing a string bos_token.")
    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError(f"HF tokenizer {hf_tokenizer_id!r} is missing a string eos_token.")
    if not isinstance(pad_token, str) or not pad_token:
        raise ValueError(f"HF tokenizer {hf_tokenizer_id!r} is missing a string pad_token.")

    bos_id_raw = hf_tokenizer.convert_tokens_to_ids(bos_token)
    eos_id_raw = hf_tokenizer.convert_tokens_to_ids(eos_token)
    pad_id_raw = hf_tokenizer.convert_tokens_to_ids(pad_token)
    if not isinstance(bos_id_raw, int):
        raise TypeError(f"Expected int bos_token_id for {bos_token!r}, got {type(bos_id_raw).__name__}.")
    if not isinstance(eos_id_raw, int):
        raise TypeError(f"Expected int eos_token_id for {eos_token!r}, got {type(eos_id_raw).__name__}.")
    if not isinstance(pad_id_raw, int):
        raise TypeError(f"Expected int pad_token_id for {pad_token!r}, got {type(pad_id_raw).__name__}.")

    bos_id = bos_id_raw
    eos_id = eos_id_raw
    pad_id = pad_id_raw

    max_id = _hf_tokenizer_max_id(hf_tokenizer)
    vocab_size = max_id + 1
    token_dtype: TokenDTypeName = "uint16" if vocab_size <= 2**16 else "uint32"

    added_vocab_raw = hf_tokenizer.get_added_vocab()
    added_vocab: dict[str, int] | None = None
    if added_vocab_raw:
        added_vocab = {str(token): int(token_id) for token, token_id in added_vocab_raw.items()}

    return TokenizerMeta(
        tokenizer="gpt-oss",
        vocab_size=vocab_size,
        token_dtype=token_dtype,
        encoding=None,
        hf_tokenizer_id=hf_tokenizer_id,
        bos_token=str(bos_token),
        bos_token_id=int(bos_id),
        eos_token=str(eos_token),
        eos_token_id=int(eos_id),
        pad_token=str(pad_token),
        pad_token_id=int(pad_id),
        special_tokens=added_vocab,
    )


@lru_cache(maxsize=None)
def _get_tiktoken_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def gpt2_meta(*, encoding: str = "gpt2") -> TokenizerMeta:
    enc = _get_tiktoken_encoding(encoding)
    eot = enc.eot_token
    if eot is None:
        raise ValueError(f"tiktoken encoding {encoding!r} has no eot_token")
    eos_id = int(eot)
    base_vocab_size = int(getattr(enc, "n_vocab", eos_id + 1))
    bos_id = base_vocab_size

    # For compatibility with NanoGPT GPT-2 defaults: 50257 rounded up for efficiency.
    required_vocab_size = max(base_vocab_size, bos_id + 1, eos_id + 1)
    vocab_size_rounded = int(((required_vocab_size + 63) // 64) * 64)
    if vocab_size_rounded < required_vocab_size:
        raise ValueError(f"Invalid rounded vocab size: {vocab_size_rounded} < {required_vocab_size}")

    return TokenizerMeta(
        tokenizer="gpt2",
        vocab_size=vocab_size_rounded,
        token_dtype="uint16",
        encoding=encoding,
        hf_tokenizer_id=None,
        bos_token=GPT2_BOS_TOKEN,
        bos_token_id=bos_id,
        eos_token=GPT2_EOS_TOKEN,
        eos_token_id=eos_id,
        pad_token=GPT2_EOS_TOKEN,
        pad_token_id=eos_id,
        special_tokens={GPT2_BOS_TOKEN: bos_id, GPT2_EOS_TOKEN: eos_id},
    )


def byteoss_meta() -> TokenizerMeta:
    return TokenizerMeta(
        tokenizer="byteoss",
        vocab_size=BYTEOSS_VOCAB_SIZE,
        token_dtype="uint32",
        encoding=None,
        hf_tokenizer_id=None,
        bos_token=BYTEOSS_BOS_TOKEN,
        bos_token_id=BYTEOSS_BOS_TOKEN_ID,
        eos_token=BYTEOSS_EOS_TOKEN,
        eos_token_id=BYTEOSS_EOS_TOKEN_ID,
        pad_token=BYTEOSS_PAD_TOKEN,
        pad_token_id=BYTEOSS_PAD_TOKEN_ID,
        special_tokens=BYTEOSS_SPECIAL_TOKEN_TO_ID,
    )


def tokenizer_meta(
    tokenizer: TokenizerName, *, encoding: str = "gpt2", hf_tokenizer_id: str = DEFAULT_GPT_OSS_TOKENIZER_ID
) -> TokenizerMeta:
    if tokenizer == "gpt2":
        return gpt2_meta(encoding=encoding)
    if tokenizer == "byteoss":
        return byteoss_meta()
    if tokenizer == "gpt-oss":
        return gpt_oss_meta(hf_tokenizer_id=hf_tokenizer_id)
    raise ValueError(f"Unknown tokenizer: {tokenizer!r}")


def numpy_token_dtype(token_dtype: TokenDTypeName) -> np.dtype[np.generic]:
    if token_dtype == "uint16":
        return np.dtype(np.uint16)
    if token_dtype == "uint32":
        return np.dtype(np.uint32)
    raise ValueError(f"Unknown token dtype: {token_dtype!r}")
