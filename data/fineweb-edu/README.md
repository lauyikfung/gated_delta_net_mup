# fineweb-edu dataset

After running `fineweb-edu.py` (preprocess) we get:

- sharded `.bin` files: `fineweb_val_000000.bin`, `fineweb_train_000001.bin`, ...
- `meta.pkl` (tokenizer + datafile header metadata, used by `train_adam_finewebedu.py`)

Version presets (Hugging Face sample sizes):

- `--version=10B`  -> `data/fineweb-edu/fineweb-edu10B/` by default
- `--version=100B` -> `data/fineweb-edu/fineweb-edu100B/` by default
- `--version=350B` -> `data/fineweb-edu/fineweb-edu350B/` by default

Token boundaries:

- For `--tokenizer=gpt2`, each document is wrapped with BOS...EOS
  (`<|beginoftext|>...<|endoftext|>`). This repo adds a distinct GPT-2 BOS token id so boundaries between
  documents become adjacent EOS/BOS tokens (`<|endoftext|><|beginoftext|>`).
- During `.bin` writing, we optionally apply aligned streaming packing ("Fit or Cut") to avoid short
  orphan tail fragments at the start of a training block.
  - Disabled by default: `--no-fit_or_cut` (enable with `--fit_or_cut`).
  - Controlled by `--block_size` (model context window) and `--fit_or_cut_threshold` (minimum viable
    fragment length).
  - With `TrainAdamConfigBase.data_loader=stream` and `stream_packing=auto` (default), disabling
    preprocessing-time packing selects runtime Fit-or-Cut packing in the `DataLoader`.

Tokenizer options:

- Default: `--tokenizer=gpt2` (tiktoken GPT-2 BPE, `uint16` tokens)
  - By default, special token strings in document text are treated as literal text (no special-token parsing).
    Use `--allow_special_tokens` to parse them into special token IDs.
- Alternative: `--tokenizer=gpt-oss` (Hugging Face tokenizer, default: `openai/gpt-oss-120b`, `uint32` tokens)
  - Uses the official GPT-OSS BPE vocabulary (~200k tokens) and wraps each document with
    `<|startoftext|>...<|return|>`.
  - By default, document text is encoded without special-token parsing so raw
    `<|startoftext|>`/`<|return|>`/`<|endoftext|>` strings are treated as literal text. Use
    `--allow_special_tokens` to parse special token strings inside document text.
  - Override the tokenizer repo with `--hf_tokenizer_id=...`.
- Alternative: `--tokenizer=byteoss` (UTF-8 bytes + GPT-OSS special tokens, `uint32` tokens)
  - By default, document text is encoded as raw UTF-8 bytes (no `<|...|>` special-token parsing), so raw
    `<|startoftext|>`/`<|return|>`/`<|endoftext|>` strings are treated as literal text. Use
    `--allow_special_tokens` to parse special token strings inside document text.
  - Training scripts default to `byteoss_vocab=compact`, remapping special token IDs so the model uses
    `vocab_size=277` (smaller embedding/LM head). Use `--byteoss_vocab=sparse` to keep `vocab_size=200019`.

Datafile format:

- Each shard is a `.bin` file containing a 256-int32 header (1024 bytes) followed by a flat token stream.

references:

- FineWeb-EDU dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
