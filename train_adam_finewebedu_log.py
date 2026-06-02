import importlib
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from utils.byteoss_vocab import BYTEOSS_COMPACT_VOCAB_SIZE, BYTEOSS_SPARSE_VOCAB_SIZE, byteoss_compactify_ids_inplace
from dataloader import (
    FitOrCutTokenStreamDataloader,
    FitOrCutTokenStreamDataloaderConfig,
    TokenStreamDataloader,
    TokenStreamDataloaderConfig,
)
from utils.optim_utils import get_optimizer_param_groups
from utils.gdn_adam_utils import get_gdn_adam_param_groups
from utils.reproducibility import fold_in_seed
from utils.seed_utils import seed_everything
from utils.checkpoint_utils import prune_old_checkpoints, update_best_checkpoint
from utils.dist_utils import broadcast_string
from utils.torch_compile import maybe_torch_compile
from train_adam_config import FinewebEduTrainConfig, load_train_config
torch.set_num_threads(1)
# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M)
_defaults = FinewebEduTrainConfig.model_construct()

# I/O
data_path = _defaults.data_path
out_dir = _defaults.out_dir
resume_dir = _defaults.resume_dir
eval_interval = _defaults.eval_interval
log_interval = _defaults.log_interval
eval_iters = _defaults.eval_iters
eval_only = _defaults.eval_only
save_checkpoints = _defaults.save_checkpoints
always_save_checkpoint = _defaults.always_save_checkpoint
keep_last_checkpoints = _defaults.keep_last_checkpoints
init_from = _defaults.init_from

# wandb logging
wandb_log = _defaults.wandb_log
wandb_project = _defaults.wandb_project
wandb_run_name = _defaults.wandb_run_name

# data
dataset = _defaults.dataset
tokenizer = _defaults.tokenizer
byteoss_vocab = _defaults.byteoss_vocab
gradient_accumulation_steps = _defaults.gradient_accumulation_steps
batch_size = _defaults.batch_size
global_batch_size = _defaults.global_batch_size
block_size = _defaults.block_size
data_loader = _defaults.data_loader
stream_packing = _defaults.stream_packing
fit_or_cut_threshold = _defaults.fit_or_cut_threshold
ignore_doc_start_loss = _defaults.ignore_doc_start_loss

# reproducibility
seed = _defaults.seed
deterministic = _defaults.deterministic
bitwise_deterministic = _defaults.bitwise_deterministic
data_seed = _defaults.data_seed
eval_seed = _defaults.eval_seed
data_rng_mode = _defaults.data_rng_mode

# model
num_hidden_layers = _defaults.num_hidden_layers
num_attention_heads = _defaults.num_attention_heads
hidden_size = _defaults.hidden_size
head_dim = _defaults.head_dim
tpa_kvrank = _defaults.tpa_kvrank
tpa_qrank = _defaults.tpa_qrank
dropout = _defaults.dropout
bias = _defaults.bias
using_groupnorm = _defaults.using_groupnorm

# muP
mup = _defaults.mup
mymup = _defaults.mymup
hidden_size_base = _defaults.hidden_size_base
embedding_lr_multiplier = _defaults.embedding_lr_multiplier

# KV shifting
use_k_shift = _defaults.use_k_shift
use_v_shift = _defaults.use_v_shift

# initialization / normalization knobs
embedding_init_std = _defaults.embedding_init_std
hidden_init_std_factor = _defaults.hidden_init_std_factor
use_qk_rmsnorm = _defaults.use_qk_rmsnorm
rope_ratio = _defaults.rope_ratio
p_tie_mode = _defaults.p_tie_mode
p_head_dim = _defaults.p_head_dim

# optimizer
optimizer_name = _defaults.optimizer_name
learning_rate_base = _defaults.learning_rate_base
max_iters = _defaults.max_iters
weight_decay = _defaults.weight_decay
beta1 = _defaults.beta1
beta2 = _defaults.beta2
grad_clip = _defaults.grad_clip
zero_stage = _defaults.zero_stage

# learning rate decay settings
decay_lr = _defaults.decay_lr
warmup_iters = _defaults.warmup_iters
lr_decay_iters = _defaults.lr_decay_iters
min_lr_base = _defaults.min_lr_base

# DDP settings
backend = _defaults.backend

# scheduler
schedule = _defaults.schedule

# model variants
model_type = _defaults.model_type
num_key_value_heads = _defaults.num_key_value_heads

# system
device = _defaults.device
dtype = _defaults.dtype
compile = _defaults.compile
scale_attn_by_inverse_layer_idx = _defaults.scale_attn_by_inverse_layer_idx

# Pydantic config + NanoGPT-style config-file/CLI overrides.
_config, extra_config = load_train_config(FinewebEduTrainConfig, sys.argv[1:])
config = _config.model_dump()  # useful for logging
for k, v in config.items():
    globals()[k] = v
# Some CUDA determinism also requires setting CUBLAS_WORKSPACE_CONFIG before CUDA context init.
if deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
# -----------------------------------------------------------------------------
model_file = importlib.import_module(f"model.{model_type}")
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT


def get_num_params(self, non_embedding=False):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params


# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get("SLURM_JOB_ID", "0")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    print(
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, "
        f"RANK: {os.environ.get('RANK')}, "
        f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}"
    )
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    world_size = 1

requested_global_batch_size: int | None = int(global_batch_size) if global_batch_size is not None else None
if requested_global_batch_size is not None:
    denom = int(batch_size) * int(world_size)
    if requested_global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size={requested_global_batch_size} must be divisible by "
            f"batch_size({int(batch_size)}) * world_size({int(world_size)}) = {denom}."
        )
    gradient_accumulation_steps = requested_global_batch_size // denom

effective_global_batch_size: int = int(batch_size) * int(gradient_accumulation_steps) * int(world_size)
if master_process and requested_global_batch_size is not None:
    print(
        "Derived gradient_accumulation_steps from global_batch_size: "
        f"global_batch_size={effective_global_batch_size} batch_size={int(batch_size)} "
        f"world_size={int(world_size)} gradient_accumulation_steps={int(gradient_accumulation_steps)}"
    )

if zero_stage not in (0, 1):
    raise ValueError(f"zero_stage must be 0 or 1, got {zero_stage}")
if zero_stage == 1 and not ddp:
    raise ValueError("zero_stage=1 requires DDP (launch with torchrun)")

# Calculate total tokens in billions
tokens_per_iter = effective_global_batch_size * block_size
total_tokens_B = tokens_per_iter * max_iters / (1000**3)

# Add after the initial variable declarations
tokens_trained = 0  # track total tokens trained

# Initialize random seed and torch settings
seed_everything(seed + seed_offset, deterministic=deterministic)
# Configure TF32 behavior (PyTorch 2.9+ uses fp32_precision; older versions use allow_tf32).
try:
    if bitwise_deterministic:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "none"
    else:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
except AttributeError:
    torch.backends.cuda.matmul.allow_tf32 = not bitwise_deterministic
    torch.backends.cudnn.allow_tf32 = not bitwise_deterministic
if bitwise_deterministic:
    torch.set_float32_matmul_precision("highest")
    # Force deterministic SDPA kernel selection (slower, but stable).
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
device_type = torch.device(device).type  # for later use in torch.autocast / torch.amp.GradScaler
# Note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
if device_type == "cpu" or dtype == "float32":
    ctx = nullcontext()
else:
    ctx = torch.autocast(device_type=device_type, dtype=ptdtype)

# Poor man's data loader
data_dir = os.path.join(data_path, dataset)
meta_path = os.path.join(data_dir, "meta.pkl")
meta_token_dtype_name = "uint16" if tokenizer == "gpt2" else "uint32"
meta_for_loader: dict[str, object] = {}
header_bytes = 1024
bos_token_id: int | None = None
eos_token_id: int | None = None
meta_packing_enabled: bool | None = None
meta_packing_block_size: int | None = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta_for_loader = pickle.load(f)
    meta_tokenizer = meta_for_loader.get("tokenizer")
    if meta_tokenizer is not None and str(meta_tokenizer) != tokenizer:
        raise ValueError(f"Dataset tokenizer={meta_tokenizer!r} does not match config tokenizer={tokenizer!r}.")
    meta_token_dtype_name = str(meta_for_loader.get("token_dtype", meta_token_dtype_name))
    datafile_info = meta_for_loader.get("datafile", {})
    if isinstance(datafile_info, dict):
        header_bytes = int(datafile_info.get("header_bytes", header_bytes))
    packing_info = meta_for_loader.get("packing", {})
    if isinstance(packing_info, dict):
        raw_enabled = packing_info.get("enabled")
        if isinstance(raw_enabled, bool):
            meta_packing_enabled = raw_enabled
        raw_block_size = packing_info.get("block_size")
        if raw_block_size is not None:
            meta_packing_block_size = int(raw_block_size)
    raw_bos_token_id = meta_for_loader.get("bos_token_id")
    if raw_bos_token_id is not None:
        bos_token_id = int(raw_bos_token_id)
    raw_eos_token_id = meta_for_loader.get("eos_token_id")
    if raw_eos_token_id is not None:
        eos_token_id = int(raw_eos_token_id)
    if tokenizer == "gpt2" and bos_token_id is not None and eos_token_id is not None and bos_token_id == eos_token_id:
        raise ValueError(
            "GPT-2 BOS/EOS are identical in meta.pkl; re-run preprocessing with the updated tokenizer "
            "so BOS and EOS are distinct."
        )

if meta_token_dtype_name == "uint16":
    token_dtype = np.uint16
elif meta_token_dtype_name == "uint32":
    token_dtype = np.uint32
else:
    raise ValueError(f"Unsupported token_dtype={meta_token_dtype_name!r} in {meta_path} (expected 'uint16'|'uint32').")


def _load_datafile_tokens(path: str) -> np.memmap:
    file_bytes = int(os.path.getsize(path))
    itemsize = int(np.dtype(token_dtype).itemsize)
    if header_bytes < 0 or header_bytes % itemsize != 0:
        raise ValueError(f"Invalid header_bytes={header_bytes} for token dtype itemsize={itemsize} in {path}.")
    payload_bytes = file_bytes - header_bytes
    if payload_bytes <= 0 or payload_bytes % itemsize != 0:
        raise ValueError(
            f"Invalid datafile size for {path}: file_bytes={file_bytes} header_bytes={header_bytes} itemsize={itemsize}."
        )
    token_count = payload_bytes // itemsize
    return np.memmap(path, dtype=token_dtype, mode="r", offset=header_bytes, shape=(token_count,))


train_file_list = sorted(
    list(
        [
            file_name
            for file_name in os.listdir(data_dir)
            if file_name.endswith(".bin") and file_name.startswith("fineweb_train")
        ]
    )
)
train_data_list = [_load_datafile_tokens(os.path.join(data_dir, file_name)) for file_name in train_file_list]
val_data = _load_datafile_tokens(os.path.join(data_dir, "fineweb_val_000000.bin"))
train_data_rng = torch.Generator(device="cpu")
train_data_rng.manual_seed(data_seed + ddp_rank)
train_py_rng = random.Random(data_seed + ddp_rank)
eval_data_rng = torch.Generator(device="cpu")
eval_data_rng.manual_seed(eval_seed)
eval_py_rng = random.Random(eval_seed)
stateless_train_rng = torch.Generator(device="cpu")
stateless_eval_rng = torch.Generator(device="cpu")
use_token_stream_dataloader = data_loader == "stream"
_stream_packing_mode: str | None = None
if use_token_stream_dataloader:
    if stream_packing == "auto":
        if meta_packing_enabled is True:
            _stream_packing_mode = "static"
        elif meta_packing_enabled is False:
            _stream_packing_mode = "dynamic"
        else:
            _stream_packing_mode = "static"
    elif stream_packing in ("static", "dynamic"):
        _stream_packing_mode = stream_packing
    else:
        raise ValueError(f"Unknown stream_packing={stream_packing!r} (expected 'auto'|'static'|'dynamic').")

train_stream: TokenStreamDataloader | FitOrCutTokenStreamDataloader | None = None
if use_token_stream_dataloader and _stream_packing_mode == "static":
    if meta_packing_enabled is False:
        raise ValueError(
            "stream_packing='static' requires preprocessing-time Fit-or-Cut packing (meta.pkl packing.enabled=True). "
            "Re-run preprocessing with --fit_or_cut, or set stream_packing='dynamic'."
        )
    if meta_packing_enabled and meta_packing_block_size is not None and int(meta_packing_block_size) != int(block_size):
        raise ValueError(
            "Dataset was prepared with Fit-or-Cut aligned packing for a different context window: "
            f"meta.pkl packing.block_size={int(meta_packing_block_size)} vs config block_size={int(block_size)}. "
            "Re-run preprocessing with the new block size, or disable preprocessing-time packing and pack at runtime."
        )
    train_stream = TokenStreamDataloader(
        shards=train_data_list,
        config=TokenStreamDataloaderConfig(
            batch_size=batch_size,
            block_size=block_size,
            seed=data_seed,
            shuffle=True,
            rank=ddp_rank,
            world_size=world_size,
            bos_token_id=bos_token_id if ignore_doc_start_loss else None,
        ),
        device=device,
    )
elif use_token_stream_dataloader and _stream_packing_mode == "dynamic":
    if meta_packing_enabled is True:
        raise ValueError(
            "Dynamic stream packing requires a raw BOS...EOS document stream (meta.pkl packing.enabled=False). "
            "Re-run preprocessing with --no-fit_or_cut to store raw documents, or set stream_packing='static'."
        )
    if bos_token_id is None or eos_token_id is None:
        raise ValueError("Dynamic stream packing requires bos_token_id and eos_token_id in meta.pkl.")
    if int(fit_or_cut_threshold) >= int(block_size):
        raise ValueError(
            "fit_or_cut_threshold must be < block_size for dynamic packing, got "
            f"fit_or_cut_threshold={int(fit_or_cut_threshold)} block_size={int(block_size)}."
        )
    train_stream = FitOrCutTokenStreamDataloader(
        shards=train_data_list,
        config=FitOrCutTokenStreamDataloaderConfig(
            batch_size=batch_size,
            block_size=block_size,
            fit_or_cut_threshold=int(fit_or_cut_threshold),
            doc_bos_token_id=bos_token_id,
            doc_eos_token_id=eos_token_id,
            seed=data_seed,
            shuffle=True,
            rank=ddp_rank,
            world_size=world_size,
            bos_token_id=bos_token_id if ignore_doc_start_loss else None,
        ),
        device=device,
    )

_byteoss_compact_vocab_validated = False


def _maybe_byteoss_compactify_ids_inplace(ids: torch.Tensor) -> None:
    global _byteoss_compact_vocab_validated
    if tokenizer != "byteoss" or byteoss_vocab != "compact":
        return
    validate = not _byteoss_compact_vocab_validated
    byteoss_compactify_ids_inplace(ids, validate=validate)
    if validate:
        _byteoss_compact_vocab_validated = True


def get_batch(
    split,
    *,
    rng: torch.Generator | None = None,
    py_rng: random.Random | None = None,
    batch_id: int | None = None,
    base_seed: int | None = None,
    rank: int | None = None,
):
    if data_rng_mode == "stateful":
        if rng is None:
            raise ValueError("stateful data_rng_mode requires rng=")
        batch_rng = rng
        if split == "train":
            if py_rng is None:
                raise ValueError("stateful data_rng_mode requires py_rng= for split='train'")
            data = py_rng.choice(train_data_list)
        else:
            data = val_data
    elif data_rng_mode == "stateless":
        if batch_id is None:
            raise ValueError("stateless data_rng_mode requires batch_id=")
        base = base_seed if base_seed is not None else (data_seed if split == "train" else eval_seed)
        eff_rank = rank if rank is not None else (ddp_rank if split == "train" else 0)
        batch_rng = (
            stateless_train_rng if (base_seed is None and rank is None and split == "train") else stateless_eval_rng
        )
        batch_rng.manual_seed(fold_in_seed(base, eff_rank, batch_id))
        if split == "train":
            file_idx = torch.randint(len(train_data_list), (1,), generator=batch_rng).item()
            data = train_data_list[file_idx]
        else:
            data = val_data
    else:
        raise ValueError(f"Unknown data_rng_mode={data_rng_mode!r}")

    ix = torch.randint(len(data) - block_size, (batch_size,), generator=batch_rng)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if ignore_doc_start_loss and bos_token_id is not None:
        y[y == bos_token_id] = -1

    _maybe_byteoss_compactify_ids_inplace(x)
    _maybe_byteoss_compactify_ids_inplace(y)

    if device_type == "cuda":
        # pin arrays x, y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


train_data_rng_state_for_batch: torch.ByteTensor | None = None
train_py_rng_state_for_batch: object | None = None
train_stream_state_for_batch: dict[str, int] | None = None
eval_data_rng_state_for_eval: torch.ByteTensor | None = None
eval_py_rng_state_for_eval: object | None = None


def _rng_state_filename() -> str:
    # Hugging Face Trainer convention: `rng_state.pth` (single process) or `rng_state_{rank}.pth` under DDP.
    return f"rng_state_{ddp_rank}.pth" if ddp else "rng_state.pth"


def _save_rng_state(ckpt_dir: str) -> None:
    payload: dict[str, object] = {
        "iter_num": iter_num,
        "world_size": world_size,
        "ddp_rank": ddp_rank,
        "data_rng_mode": data_rng_mode,
        "data_loader": data_loader,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if device_type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)

    if use_token_stream_dataloader:
        if train_stream_state_for_batch is None:
            raise RuntimeError("Missing train_stream_state_for_batch for stream data loader")
        payload["train_stream_state_for_batch"] = train_stream_state_for_batch
        if master_process and data_rng_mode == "stateful":
            if eval_data_rng_state_for_eval is None or eval_py_rng_state_for_eval is None:
                raise RuntimeError("Missing eval RNG state for stateful RNG mode")
            payload["eval_data_rng_state_for_eval"] = eval_data_rng_state_for_eval
            payload["eval_py_rng_state_for_eval"] = eval_py_rng_state_for_eval
    elif data_rng_mode == "stateful":
        if train_data_rng_state_for_batch is None or train_py_rng_state_for_batch is None:
            raise RuntimeError("Missing train RNG state for stateful RNG mode")
        payload["train_data_rng_state_for_batch"] = train_data_rng_state_for_batch
        payload["train_py_rng_state_for_batch"] = train_py_rng_state_for_batch
        if master_process:
            if eval_data_rng_state_for_eval is None or eval_py_rng_state_for_eval is None:
                raise RuntimeError("Missing eval RNG state for stateful RNG mode")
            payload["eval_data_rng_state_for_eval"] = eval_data_rng_state_for_eval
            payload["eval_py_rng_state_for_eval"] = eval_py_rng_state_for_eval

    torch.save(payload, os.path.join(ckpt_dir, _rng_state_filename()))


def _get_train_batch_stateful():
    global train_data_rng_state_for_batch, train_py_rng_state_for_batch
    train_data_rng_state_for_batch = train_data_rng.get_state()
    train_py_rng_state_for_batch = train_py_rng.getstate()
    return get_batch("train", rng=train_data_rng, py_rng=train_py_rng)


def _get_train_batch_stream():
    global train_stream_state_for_batch
    if train_stream is None:
        raise RuntimeError("stream data loader is not initialized")
    train_stream_state_for_batch = train_stream.state_dict()
    x, y = train_stream.next_batch()
    _maybe_byteoss_compactify_ids_inplace(x)
    _maybe_byteoss_compactify_ids_inplace(y)
    return x, y


def _checkpoint_dir(run_dir: str, step: int) -> str:
    return os.path.join(run_dir, f"checkpoint-{step}")


def _resolve_resume_checkpoint_dir(path: str) -> str:
    base = os.path.basename(path.rstrip("/"))
    if base.startswith("checkpoint-"):
        return path

    candidates: list[tuple[int, str]] = []
    for name in os.listdir(path):
        if not name.startswith("checkpoint-"):
            continue
        _, _, suffix = name.partition("-")
        if not suffix.isdigit():
            continue
        candidates.append((int(suffix), os.path.join(path, name)))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories found under {path!r}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
# Build model_args with defaults that can be overridden by config globals
model_args = {
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "hidden_size": hidden_size,
    "block_size": block_size,
    "bias": bias,
    "head_dim": head_dim,
    "tpa_kvrank": tpa_kvrank,
    "tpa_qrank": tpa_qrank,
    "using_groupnorm": using_groupnorm,
    "vocab_size": None,
    "dropout": dropout,
    "scale_attn_by_inverse_layer_idx": scale_attn_by_inverse_layer_idx,
    # Init/normalization knobs (with sensible defaults)
    "embedding_init_std": embedding_init_std,
    "hidden_init_std_factor": hidden_init_std_factor,
    "use_qk_rmsnorm": use_qk_rmsnorm,
    "rope_ratio": rope_ratio,
    "use_k_shift": use_k_shift,
    "use_v_shift": use_v_shift,
    "p_tie_mode": p_tie_mode,
    "p_head_dim": p_head_dim,
    # muP knobs (forward/logits scaling + optimizer LR grouping)
    "mup": mup,
    "mymup": mymup,
    "hidden_size_base": hidden_size_base,
    "embedding_lr_multiplier": embedding_lr_multiplier,
}

if "gqa" in model_type:
    model_args["num_key_value_heads"] = num_key_value_heads

# Pass through any GRAPE-specific hyperparameters provided via config files.
if "grape" in model_type:
    for key, value in extra_config.items():
        if key.startswith("grape_"):
            model_args[key] = value

# Pass through any key-gated additive bias hyperparameters provided via config files.
if "keygated" in model_type:
    for key, value in extra_config.items():
        if key.startswith("keygated_"):
            model_args[key] = value

# Pass through any query-gated additive bias hyperparameters provided via config files.
if "querygated" in model_type or "querykeygated" in model_type:
    for key, value in extra_config.items():
        if key.startswith("querygated_"):
            model_args[key] = value

# Pass through any query+key-gated additive bias hyperparameters provided via config files.
if "querykeygated" in model_type:
    for key, value in extra_config.items():
        if key.startswith("querykeygated_"):
            model_args[key] = value

for key, value in extra_config.items():
    if key not in model_args:
        model_args[key] = value

if "delta_net" in model_type:
    for key, value in extra_config.items():
        if key.startswith("delta_"):
            model_args[key] = value

if "gated_deltanet" in model_type:
    for key, value in extra_config.items():
        if key.startswith("gated_deltanet_"):
            model_args[key] = value

resume_checkpoint_dir: str | None = None
resume_run_dir: str | None = None

if init_from == "scratch":
    # Init a new model from scratch
    print("Initializing a new model from scratch")
    # Determine the vocab size we'll use for from-scratch training
    if tokenizer == "gpt2":
        default_vocab_size = 50304
        model_vocab_size = meta_vocab_size if meta_vocab_size is not None else default_vocab_size
    elif tokenizer == "byteoss":
        default_vocab_size = BYTEOSS_COMPACT_VOCAB_SIZE if byteoss_vocab == "compact" else BYTEOSS_SPARSE_VOCAB_SIZE
        model_vocab_size = (
            BYTEOSS_COMPACT_VOCAB_SIZE if byteoss_vocab == "compact" else (meta_vocab_size or default_vocab_size)
        )
    elif tokenizer == "gpt-oss":
        default_vocab_size = 200019
        model_vocab_size = meta_vocab_size if meta_vocab_size is not None else default_vocab_size
    else:
        raise ValueError(f"Unsupported tokenizer={tokenizer!r}")

    if meta_vocab_size is None:
        print(f"defaulting to vocab_size of {tokenizer} to {default_vocab_size}")
    elif tokenizer == "byteoss" and byteoss_vocab == "compact":
        print(f"dataset vocab_size={meta_vocab_size} but using compact model vocab_size={model_vocab_size}")

    model_args["vocab_size"] = model_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {resume_dir}")
    resume_checkpoint_dir = _resolve_resume_checkpoint_dir(resume_dir)
    resume_run_dir = os.path.dirname(resume_checkpoint_dir.rstrip("/"))
    print(f"Loading checkpoint from {resume_checkpoint_dir}")
    config = GPTConfig.from_json_file(os.path.join(resume_checkpoint_dir, "config.json"))
    model = GPT.from_pretrained(resume_checkpoint_dir, config=config)

    # Force these config attributes to be equal otherwise we can't even resume training
    # The rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["num_hidden_layers", "num_attention_heads", "hidden_size", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(config, k)
    model.tie_weights()
    if tokenizer == "byteoss":
        vocab_size = int(getattr(config, "vocab_size"))
        inferred_vocab = None
        if vocab_size == int(BYTEOSS_COMPACT_VOCAB_SIZE):
            inferred_vocab = "compact"
        elif vocab_size == int(BYTEOSS_SPARSE_VOCAB_SIZE):
            inferred_vocab = "sparse"
        if inferred_vocab is not None and byteoss_vocab != inferred_vocab:
            raise ValueError(
                f"Checkpoint vocab_size={vocab_size} implies byteoss_vocab={inferred_vocab!r}, "
                f"but config byteoss_vocab={byteoss_vocab!r}."
            )
        if inferred_vocab is not None:
            byteoss_vocab = inferred_vocab
else:
    raise ValueError(
        f"Unsupported init_from={init_from!r}. Expected 'scratch' or 'resume' "
        "(resume from a directory produced by this repo's training scripts)."
    )
# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size  # so that the checkpoint will have the right value

# Persist tokenizer metadata into the checkpoint config for downstream inference.
model.config.tokenizer = tokenizer
if tokenizer == "gpt2":
    encoding_name = meta_for_loader.get("encoding")
    if encoding_name is not None:
        model.config.encoding = str(encoding_name)
elif tokenizer == "gpt-oss":
    hf_tokenizer_id = meta_for_loader.get("hf_tokenizer_id")
    if hf_tokenizer_id is not None:
        model.config.hf_tokenizer_id = str(hf_tokenizer_id)
elif tokenizer == "byteoss":
    model.config.byteoss_vocab = byteoss_vocab

# Persist the NanoGPT-Next model module name for evaluation harness integrations.
model.config.nanogpt_next_model_type = model_type

model.to(device)

# Now calculate non-embedding parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000  # convert to millions

# Update wandb run name and out_dir if not resuming
mup = bool(getattr(model.config, "mup", mup))
mup_midfix = "-mup" if mup else ""
if init_from != "resume":
    # Update wandb run name
    wandb_run_name = f"W{hidden_size}_LR{learning_rate_base}_{optimizer_name}_{model_type}{mup_midfix}_T_{total_tokens_B:.2f}B_{current_date}_log"
    # Update output directory
    out_dir = f"output/out_{model_type}_W{hidden_size}_{optimizer_name}_{model_type}{mup_midfix}_LR{learning_rate_base}_T_{total_tokens_B:.2f}B_time_{current_date}"
else:
    # Default to writing checkpoints next to the resume checkpoint directory.
    # Users can still override `out_dir` explicitly via config/CLI.
    if resume_run_dir is not None and out_dir == _defaults.out_dir:
        out_dir = resume_run_dir
    wandb_run_name = f"W{hidden_size}_LR{learning_rate_base}_{optimizer_name}_{model_type}{mup_midfix}_T_{total_tokens_B:.2f}B_{current_date}_log"

# Keep generated identifiers consistent across ranks (local clocks can differ by seconds).
if ddp:
    comm_device = device if backend == "nccl" else "cpu"
    out_dir = broadcast_string(out_dir, src=0, device=comm_device)
    wandb_run_name = broadcast_string(wandb_run_name, src=0, device=comm_device)
# Now create the output directory
if master_process:
    os.makedirs(out_dir, exist_ok=True)
if ddp:
    dist.barrier()

# Initialize a GradScaler. If enabled=False, scaler is a no-op
try:
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16" and device_type == "cuda"))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16" and device_type == "cuda"))

# Optimizer
# muP: for resume, prefer values saved in config.json (missing => keep current config/defaults)
mup = bool(getattr(model.config, "mup", mup))
hidden_size_base = int(getattr(model.config, "hidden_size_base", hidden_size_base))
embedding_lr_multiplier = float(getattr(model.config, "embedding_lr_multiplier", embedding_lr_multiplier))
model_hidden_size = float(getattr(model.config, "hidden_size", hidden_size))
hidden_lr_mult = (float(hidden_size_base) / model_hidden_size) if mup else 1.0
embedding_lr_mult = float(embedding_lr_multiplier) if mup else 1.0

# Whether this run uses a GDN model — needed here for muP param group selection.
_is_gdn_model = "gdn" in model_type

if _is_gdn_model and mup:
    # GDN+muP: use dedicated param groups to give A_log/dt_bias lr=1.0 (not reduced)
    # and zero weight decay, per mup_adamw.md.
    param_groups = get_gdn_adam_param_groups(
        model,
        weight_decay,
        hidden_lr_mult=hidden_lr_mult,
        embedding_lr_mult=embedding_lr_mult,
    )
else:
    param_groups = get_optimizer_param_groups(
        model,
        weight_decay,
        split_embeddings=mup,
        embedding_lr_mult=embedding_lr_mult,
        hidden_lr_mult=hidden_lr_mult,
    )
use_zero1 = zero_stage == 1
if use_zero1:
    optimizer = ZeroRedundancyOptimizer(
        param_groups,
        optimizer_class=AdamW,
        lr=learning_rate_base,
        betas=(beta1, beta2),
        eps=1e-8,
    )
else:
    optimizer = AdamW(
        param_groups,
        lr=learning_rate_base,
        betas=(beta1, beta2),
        eps=1e-8,
    )
if init_from == "resume":
    if resume_checkpoint_dir is None:
        resume_checkpoint_dir = _resolve_resume_checkpoint_dir(resume_dir)

    trainer_state_path = os.path.join(resume_checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, encoding="utf-8") as f:
        trainer_state = json.load(f)
    iter_num = int(trainer_state["global_step"])
    best_val_loss = float(trainer_state.get("best_val_loss", best_val_loss))
    tokens_trained = int(trainer_state.get("tokens_trained", iter_num * int(tokens_per_iter)))

    optimizer_state_path = os.path.join(resume_checkpoint_dir, "optimizer.pt")
    optimizer_state = torch.load(
        optimizer_state_path,
        map_location=("cpu" if use_zero1 else device),
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state

    scaler_state_path = os.path.join(resume_checkpoint_dir, "scaler.pt")
    if os.path.exists(scaler_state_path):
        scaler_state = torch.load(
            scaler_state_path,
            map_location="cpu",
            weights_only=False,
        )
        scaler.load_state_dict(scaler_state)
        del scaler_state
    elif scaler.is_enabled():
        raise FileNotFoundError(f"FP16 training requires {scaler_state_path} to resume.")

    rng_state_path = os.path.join(resume_checkpoint_dir, _rng_state_filename())
    if not os.path.exists(rng_state_path):
        raise FileNotFoundError(f"Missing RNG state file: {rng_state_path}")
    rng_state = torch.load(
        rng_state_path,
        map_location="cpu",
        weights_only=False,
    )

    ckpt_iter = rng_state.get("iter_num")
    ckpt_world_size = rng_state.get("world_size")
    ckpt_rank = rng_state.get("ddp_rank")
    if ckpt_iter is not None and int(ckpt_iter) != int(iter_num):
        raise RuntimeError(
            f"RNG state iter mismatch: rng_state iter_num={ckpt_iter} but trainer_state global_step={iter_num} "
            f"({rng_state_path})"
        )
    if ckpt_world_size is not None and int(ckpt_world_size) != int(world_size):
        raise RuntimeError(
            f"RNG state WORLD_SIZE mismatch: rng_state world_size={ckpt_world_size} "
            f"but current WORLD_SIZE={world_size} ({rng_state_path})"
        )
    if ckpt_rank is not None and int(ckpt_rank) != int(ddp_rank):
        raise RuntimeError(
            f"RNG state RANK mismatch: rng_state ddp_rank={ckpt_rank} but current RANK={ddp_rank} ({rng_state_path})"
        )

    random.setstate(rng_state["python_random_state"])
    np.random.set_state(rng_state["numpy_random_state"])
    torch.set_rng_state(rng_state["torch_rng_state"])
    if device_type == "cuda":
        cuda_state = rng_state.get("cuda_rng_state")
        if cuda_state is None:
            raise RuntimeError(f"Missing cuda_rng_state in {rng_state_path}")
        torch.cuda.set_rng_state(cuda_state, device)

    if data_rng_mode == "stateful" and not use_token_stream_dataloader:
        torch_state = rng_state.get("train_data_rng_state_for_batch")
        py_state = rng_state.get("train_py_rng_state_for_batch")
        if torch_state is None or py_state is None:
            raise RuntimeError(f"Missing train RNG state in {rng_state_path}")
        train_data_rng_state_for_batch = torch_state
        train_py_rng_state_for_batch = py_state
        train_data_rng.set_state(torch_state)
        train_py_rng.setstate(py_state)
        if master_process:
            eval_torch_state = rng_state.get("eval_data_rng_state_for_eval")
            eval_py_state = rng_state.get("eval_py_rng_state_for_eval")
            if eval_torch_state is None or eval_py_state is None:
                raise RuntimeError(f"Missing eval RNG state in {rng_state_path}")
            eval_data_rng_state_for_eval = eval_torch_state
            eval_py_rng_state_for_eval = eval_py_state
            eval_data_rng.set_state(eval_torch_state)
            eval_py_rng.setstate(eval_py_state)

    if data_loader == "stream":
        state = rng_state.get("train_stream_state_for_batch")
        if state is None:
            raise RuntimeError(f"Missing train_stream_state_for_batch in {rng_state_path}")
        train_stream_state_for_batch = state
        if train_stream is None:
            raise RuntimeError("stream data loader is not initialized")
        train_stream.load_state_dict(state)
        if master_process and data_rng_mode == "stateful":
            eval_state = rng_state.get("eval_data_rng_state_for_eval")
            py_state = rng_state.get("eval_py_rng_state_for_eval")
            if eval_state is None or py_state is None:
                raise RuntimeError(f"Missing eval RNG state in {rng_state_path}")
            eval_data_rng_state_for_eval = eval_state
            eval_py_rng_state_for_eval = py_state
            eval_data_rng.set_state(eval_state)
            eval_py_rng.setstate(py_state)
# Compile the model
if compile:
    model = maybe_torch_compile(model, enabled=True, log=(print if master_process else None))

# Wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# Helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out: dict[str, float] = {}
    eval_model = model.module if ddp else model
    eval_model.eval()

    # GDN muP analysis: register forward hooks on a_proj and b_proj in first few layers.
    # Captures output std of these projections to verify the gate is not dead (a_proj≈0).
    _a_proj_stds: list[float] = []
    _b_proj_stds: list[float] = []
    _hooks: list = []
    if _is_gdn_model and wandb_log and master_process:
        _n_hooked = 0
        for _, _mod in eval_model.named_modules():
            if hasattr(_mod, 'a_proj') and hasattr(_mod, 'dt_bias'):
                def _make_hook(lst: list) -> object:
                    def _h(m, inp, out: torch.Tensor) -> None:
                        lst.append(out.detach().float().std().item())
                    return _h
                _hooks.append(_mod.a_proj.register_forward_hook(_make_hook(_a_proj_stds)))
                _hooks.append(_mod.b_proj.register_forward_hook(_make_hook(_b_proj_stds)))
                _n_hooked += 1
                if _n_hooked >= 4:  # first 4 GDN layers is sufficient
                    break

    _logit_stds: list[float] = []
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if data_rng_mode == "stateless":
                X, Y = get_batch(split, batch_id=k, base_seed=eval_seed, rank=0)
            else:
                X, Y = get_batch(split, rng=eval_data_rng, py_rng=eval_py_rng)
            with ctx:
                logits, loss = eval_model(X, Y)
            losses[k] = loss.item()
            # Collect logit std from first few val batches for muP logits-scaling validation.
            if split == "val" and k < 5 and logits is not None:
                _logit_stds.append(logits.detach().float().std(dim=-1).mean().item())
        out[split] = float(losses.mean().item())

    for h in _hooks:
        h.remove()

    if _logit_stds:
        out['logit_std'] = float(np.mean(_logit_stds))
    if _a_proj_stds:
        out['a_proj_output_std'] = float(np.mean(_a_proj_stds))
    if _b_proj_stds:
        out['b_proj_output_std'] = float(np.mean(_b_proj_stds))

    eval_model.train()
    return out


# Learning rate decay scheduler (cosine with warmup)
def get_lr(it, schedule="cosine"):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate_base * it / warmup_iters
    # 2) if it >= lr_decay_iters, return min learning rate
    if it >= lr_decay_iters:
        return min_lr_base
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1

    return min_lr_base + coeff * (learning_rate_base - min_lr_base)


# ---------------------------------------------------------------------------
# muP analysis helpers for GDN models
# ---------------------------------------------------------------------------

def _gdn_group_name(param_name: str) -> str:
    """Classify a GDN parameter by its role, for grouped muP analysis metrics."""
    leaf = param_name.split('.')[-1]
    if leaf in ('A_log', 'dt_bias'):
        return 'scalar'
    for p in ('a_proj', 'b_proj'):
        if f'.{p}.' in param_name or param_name.endswith(f'.{p}'):
            return 'gate_proj'
    for p in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'g_proj'):
        if f'.{p}.' in param_name or param_name.endswith(f'.{p}'):
            return 'main_proj'
    if 'wte' in param_name or 'lm_head' in param_name:
        return 'embedding'
    if 'conv1d' in param_name:
        return 'conv'
    if any(x in param_name for x in ('mlp', 'intermediate', 'gate_proj', 'down_proj', 'up_proj')):
        return 'mlp'
    if any(x in param_name for x in ('norm', 'ln_f')):
        return 'norm'
    return 'other'


def _log_gdn_group_stats(
    raw_model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    iter_num: int,
    lr: float,
) -> None:
    """Log per-group weight norms, gradient norms, effective LRs, and update norms."""
    if not (wandb_log and master_process):
        return

    w_sq: dict[str, float] = defaultdict(float)
    g_sq: dict[str, float] = defaultdict(float)
    group_lr_mult: dict[str, float] = {}
    scalar_logs: dict[str, float] = {}

    param_lr_mults: dict[int, float] = {
        id(p): float(pg.get('lr_mult', 1.0))
        for pg in opt.param_groups
        for p in pg['params']
    }

    for name, param in raw_model.named_parameters():
        g = _gdn_group_name(name)
        w_sq[g] += param.data.float().norm().item() ** 2
        if param.grad is not None:
            g_sq[g] += param.grad.float().norm().item() ** 2
        if g not in group_lr_mult:
            group_lr_mult[g] = param_lr_mults.get(id(param), 1.0)

        leaf = name.split('.')[-1]
        if leaf == 'A_log':
            v = param.data.float()
            scalar_logs['gdn/scalar/A_log_mean'] = v.mean().item()
            scalar_logs['gdn/scalar/A_log_std'] = v.std().item()
            if param.grad is not None:
                scalar_logs['gdn/scalar/A_log_grad_norm'] = param.grad.float().norm().item()
        elif leaf == 'dt_bias':
            v = param.data.float()
            scalar_logs['gdn/scalar/dt_bias_mean'] = v.mean().item()
            scalar_logs['gdn/scalar/dt_bias_std'] = v.std().item()
            if param.grad is not None:
                scalar_logs['gdn/scalar/dt_bias_grad_norm'] = param.grad.float().norm().item()

    log_dict: dict[str, float] = {}
    for g in sorted(set(w_sq) | set(g_sq)):
        log_dict[f'gdn/weight_norm/{g}'] = w_sq[g] ** 0.5
        if g_sq.get(g, 0.0) > 0:
            grad_norm = g_sq[g] ** 0.5
            eff_lr = lr * group_lr_mult.get(g, 1.0)
            log_dict[f'gdn/grad_norm/{g}'] = grad_norm
            log_dict[f'gdn/effective_lr/{g}'] = eff_lr
            log_dict[f'gdn/update_norm/{g}'] = eff_lr * grad_norm
            w_norm = w_sq[g] ** 0.5
            if w_norm > 0:
                log_dict[f'gdn/fractional_update/{g}'] = (eff_lr * grad_norm) / w_norm

    log_dict.update(scalar_logs)
    wandb.log(log_dict, step=iter_num)


# ---------------------------------------------------------------------------
# GDN activation + gradient dynamics logger
# ---------------------------------------------------------------------------

def _log_gdn_dynamics(iter_num: int) -> None:
    """Capture and log intermediate activation + gradient statistics for muP verification."""
    if not (wandb_log and master_process and _is_gdn_model):
        return

    import torch.nn.functional as _F

    # Use attribute-based duck-typing so this works for both gdn and gdn-mup.
    # A_log and dt_bias are unique to GatedDeltaNet layers.
    gdn_layers: list[tuple[int, object]] = [
        (i, block.attn)
        for i, block in enumerate(raw_model.transformer.h)
        if hasattr(block.attn, 'A_log') and hasattr(block.attn, 'dt_bias')
    ]
    if not gdn_layers:
        return

    n = len(gdn_layers)
    idx_set = sorted({0, n // 2, n - 1})
    sampled_gdn = [gdn_layers[i] for i in idx_set]

    captures: dict = defaultdict(dict)
    res_caps: dict[int, torch.Tensor] = {}
    hooks: list = []

    def _cap_out(blk_idx: int, key: str, *, with_grad: bool = False):
        def h(m, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[blk_idx][key] = t.detach().float()
            if with_grad and isinstance(t, torch.Tensor) and t.requires_grad:
                t.register_hook(
                    lambda g, _i=blk_idx, _k=key + '_grad':
                    captures[_i].__setitem__(_k, g.detach().float())
                )
        return h

    def _cap_in(blk_idx: int, key: str, *, with_grad: bool = False):
        def h(m, inp, out):
            t = inp[0] if isinstance(inp, tuple) else inp
            if not isinstance(t, torch.Tensor):
                return
            captures[blk_idx][key] = t.detach().float()
            if with_grad and t.requires_grad:
                t.register_hook(
                    lambda g, _i=blk_idx, _k=key + '_grad':
                    captures[_i].__setitem__(_k, g.detach().float())
                )
        return h

    for blk_idx, gdn in sampled_gdn:
        if gdn.use_short_conv:
            hooks.append(gdn.q_conv1d.register_forward_hook(
                _cap_out(blk_idx, 'q', with_grad=True)))
            hooks.append(gdn.k_conv1d.register_forward_hook(
                _cap_out(blk_idx, 'k', with_grad=True)))
            hooks.append(gdn.v_conv1d.register_forward_hook(
                _cap_out(blk_idx, 'v')))
        hooks.append(gdn.o_proj.register_forward_hook(
            _cap_in(blk_idx, 'o', with_grad=True)))
        hooks.append(gdn.o_proj.register_forward_hook(
            _cap_out(blk_idx, 'o_proj_out')))
        hooks.append(gdn.b_proj.register_forward_hook(
            _cap_out(blk_idx, 'b_proj_out')))
        block = raw_model.transformer.h[blk_idx]
        hooks.append(block.register_forward_hook(
            _cap_in(blk_idx, 'hidden_in', with_grad=True)))

    _sampled_idx = {bi for bi, _ in sampled_gdn}
    for blk_idx, gdn in gdn_layers:
        if blk_idx not in _sampled_idx and gdn.use_short_conv:
            hooks.append(gdn.q_conv1d.register_forward_hook(_cap_out(blk_idx, 'q')))
            hooks.append(gdn.k_conv1d.register_forward_hook(_cap_out(blk_idx, 'k')))

    for i, block in enumerate(raw_model.transformer.h):
        def _res(m, inp, out, _i=i):
            t = out[0] if isinstance(out, tuple) else out
            res_caps[_i] = t.detach().float()
        hooks.append(block.register_forward_hook(_res))

    was_training = raw_model.training
    raw_model.eval()
    optimizer.zero_grad(set_to_none=True)
    try:
        if data_rng_mode == "stateless":
            Xd, Yd = get_batch("val", batch_id=9999, base_seed=eval_seed, rank=0)
        else:
            _rng_st  = eval_data_rng.get_state()
            _py_st   = eval_py_rng.getstate()
            Xd, Yd   = get_batch("val", rng=eval_data_rng, py_rng=eval_py_rng)
            eval_data_rng.set_state(_rng_st)
            eval_py_rng.setstate(_py_st)
        with ctx:
            _, loss_d = raw_model(Xd, Yd)
        loss_d.backward()
    except Exception as _e:
        print(f"[_log_gdn_dynamics] diagnostic pass failed: {_e}")
        return
    finally:
        for h in hooks:
            h.remove()
        optimizer.zero_grad(set_to_none=True)
        if was_training:
            raw_model.train()

    def _rms(t: torch.Tensor) -> float:
        return float(t.float().pow(2).mean().sqrt().item())

    log_dict: dict[str, float] = {}
    agg: dict[str, list[float]] = defaultdict(list)

    for blk_idx, gdn in sampled_gdn:
        cap = captures[blk_idx]
        pfx = f"gdn/dyn/L{blk_idx}"

        if 'q' in cap and 'k' in cap:
            q, k = cap['q'], cap['k']
            qr, kr = _rms(q), _rms(k)
            log_dict[f'{pfx}/q_rms'] = qr;  agg['q_rms'].append(qr)
            log_dict[f'{pfx}/k_rms'] = kr;  agg['k_rms'].append(kr)

            BT = q.shape[0] * q.shape[1]
            q_h = q.reshape(BT, gdn.num_heads, gdn.head_k_dim)
            k_h = k.reshape(BT, gdn.num_heads, gdn.head_k_dim)
            cos = (_F.normalize(q_h, dim=-1, eps=1e-8) *
                   _F.normalize(k_h, dim=-1, eps=1e-8)).sum(-1)
            log_dict[f'{pfx}/qk_cos_mean']  = float(cos.mean().item())
            log_dict[f'{pfx}/qk_cos_abs']   = float(cos.abs().mean().item())
            agg['qk_cos_abs'].append(log_dict[f'{pfx}/qk_cos_abs'])

        if 'v' in cap:
            vr = _rms(cap['v'])
            log_dict[f'{pfx}/v_rms'] = vr;  agg['v_rms'].append(vr)

        if 'b_proj_out' in cap:
            beta = cap['b_proj_out'].sigmoid()
            log_dict[f'{pfx}/beta_mean']    = float(beta.mean().item())
            log_dict[f'{pfx}/beta_std']     = float(beta.std().item())
            log_dict[f'{pfx}/beta_gt_half'] = float((beta > 0.5).float().mean().item())
            agg['beta_mean'].append(log_dict[f'{pfx}/beta_mean'])

        if 'o' in cap:
            or_ = _rms(cap['o'])
            log_dict[f'{pfx}/o_rms'] = or_;  agg['o_rms'].append(or_)

        if 'o_proj_out' in cap:
            op = _rms(cap['o_proj_out'])
            log_dict[f'{pfx}/o_proj_out_rms'] = op;  agg['o_proj_out_rms'].append(op)

        if 'hidden_in' in cap:
            hr = _rms(cap['hidden_in'])
            log_dict[f'{pfx}/hidden_rms'] = hr;  agg['hidden_rms'].append(hr)

        if 'q_grad' in cap:
            gr = _rms(cap['q_grad'])
            log_dict[f'{pfx}/grad_q_rms'] = gr;  agg['grad_q_rms'].append(gr)

        if 'k_grad' in cap:
            gr = _rms(cap['k_grad'])
            log_dict[f'{pfx}/grad_k_rms'] = gr;  agg['grad_k_rms'].append(gr)

        if 'o_grad' in cap:
            gr = _rms(cap['o_grad'])
            log_dict[f'{pfx}/grad_o_rms'] = gr;  agg['grad_o_rms'].append(gr)

        if 'hidden_in_grad' in cap:
            gr = _rms(cap['hidden_in_grad'])
            log_dict[f'{pfx}/grad_hidden_rms'] = gr;  agg['grad_hidden_rms'].append(gr)

    _T_FRACS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    for blk_idx, gdn in gdn_layers:
        cap = captures[blk_idx]
        if 'q' not in cap or 'k' not in cap:
            continue
        q, k = cap['q'], cap['k']
        B, T, _ = q.shape
        q_h = q.reshape(B, T, gdn.num_heads, gdn.head_k_dim)
        k_h = k.reshape(B, T, gdn.num_heads, gdn.head_k_dim)
        cos_all = (_F.normalize(q_h, dim=-1, eps=1e-8) *
                   _F.normalize(k_h, dim=-1, eps=1e-8)).sum(-1)
        log_dict[f'gdn/dyn/qk_cos/L{blk_idx}'] = float(cos_all.mean().item())
        cos_t = cos_all.mean(dim=(0, 2))
        t_indices = sorted({min(int(f * T), T - 1) for f in _T_FRACS})
        for t in t_indices:
            log_dict[f'gdn/dyn/qk_cos_by_t/L{blk_idx}/t{t:04d}'] = float(cos_t[t].item())

    for i in sorted(res_caps.keys()):
        log_dict[f'gdn/dyn/residual/L{i}'] = _rms(res_caps[i])

    for key, vals in agg.items():
        if vals:
            log_dict[f'gdn/dyn/avg/{key}'] = sum(vals) / len(vals)

    d     = float(raw_model.config.hidden_size)
    sqrtd = math.sqrt(d)
    hkd   = float(gdn_layers[0][1].head_k_dim)

    def _a(key: str) -> float | None:
        vals = agg.get(key)
        return (sum(vals) / len(vals)) if vals else None

    avg_q = _a('q_rms');  avg_k = _a('k_rms')
    avg_o = _a('o_rms');  avg_go = _a('grad_o_rms')

    if avg_q is not None:
        q_hat = avg_q / math.sqrt(hkd)
        log_dict['gdn/dyn/mup/q_hat_per_elem'] = q_hat
        log_dict['gdn/dyn/mup/q_hat_x_sqrtd']  = q_hat * sqrtd
    if avg_k is not None:
        k_hat = avg_k / math.sqrt(hkd)
        log_dict['gdn/dyn/mup/k_hat_per_elem'] = k_hat
        log_dict['gdn/dyn/mup/k_hat_x_sqrtd']  = k_hat * sqrtd

    if avg_o is not None:
        log_dict['gdn/dyn/mup/o_rms']           = avg_o
        log_dict['gdn/dyn/mup/o_rms_div_sqrtd'] = avg_o / sqrtd

    if avg_go is not None:
        log_dict['gdn/dyn/mup/grad_o_rms']       = avg_go
        log_dict['gdn/dyn/mup/grad_o_x_sqrtd']   = avg_go * sqrtd
        log_dict['gdn/dyn/mup/grad_o_div_sqrtd']  = avg_go / sqrtd

    wandb.log(log_dict, step=iter_num)


# Logging
if wandb_log and master_process:
    import wandb

    wandb_config = {
        "model_args": model_args,
        "training_args": {
            "batch_size": batch_size,
            "global_batch_size": effective_global_batch_size,
            "requested_global_batch_size": requested_global_batch_size,
            "world_size": world_size,
            "block_size": block_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "tokens_per_iter": tokens_per_iter,
            "max_iters": max_iters,
            "lr_decay_iters": lr_decay_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "log_interval": log_interval,
        },
        "optimizer_args": {
            "optimizer_name": optimizer_name,
            "learning_rate_base": learning_rate_base,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "grad_clip": grad_clip,
            "decay_lr": decay_lr,
            "warmup_iters": warmup_iters,
            "min_lr_base": min_lr_base,
            "schedule": schedule,
        },
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)

# Training loop
if use_token_stream_dataloader:
    train_batch_id = None
    X, Y = _get_train_batch_stream()  # fetch the very first batch
elif data_rng_mode == "stateless":
    train_batch_id = iter_num * gradient_accumulation_steps
    X, Y = get_batch("train", batch_id=train_batch_id)  # fetch the very first batch
else:
    train_batch_id = None
    X, Y = _get_train_batch_stateful()  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
clip_time = 0
while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate_base
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_mult", 1.0)

    # Evaluate the loss on train/val sets and write checkpoints (keep DDP ranks in sync)
    if eval_only or iter_num % eval_interval == 0:
        save_latest = False
        val_improved = False
        if master_process:
            if data_rng_mode == "stateful":
                eval_data_rng_state_for_eval = eval_data_rng.get_state()
                eval_py_rng_state_for_eval = eval_py_rng.getstate()
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                eval_log: dict = {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
                # GDN muP analysis: logit statistics and gate-activation health
                if _is_gdn_model:
                    if 'logit_std' in losses:
                        eval_log['gdn/eval/logit_std'] = losses['logit_std']
                    if 'a_proj_output_std' in losses:
                        eval_log['gdn/eval/a_proj_output_std'] = losses['a_proj_output_std']
                    if 'b_proj_output_std' in losses:
                        eval_log['gdn/eval/b_proj_output_std'] = losses['b_proj_output_std']
                wandb.log(eval_log, step=iter_num)
                # Full dynamics capture: activations + gradients for muP verification.
                if _is_gdn_model:
                    _log_gdn_dynamics(iter_num)
            val_improved = losses["val"] < best_val_loss
            if val_improved:
                best_val_loss = losses["val"]
            if iter_num > 0 and (val_improved or always_save_checkpoint):
                save_latest = True

        if ddp:
            comm_device = device if backend == "nccl" else "cpu"
            save_latest_flag = torch.tensor([1 if save_latest else 0], device=comm_device)
            dist.broadcast(save_latest_flag, src=0)
            save_latest = bool(save_latest_flag.item())

        save_snapshot = iter_num % (eval_interval * 5) == 0

        optimizer_state_dict = None
        save_checkpoint = save_checkpoints and (save_latest or save_snapshot)
        if save_checkpoint:
            if use_zero1:
                optimizer.consolidate_state_dict(to=0)
            if master_process:
                optimizer_state_dict = optimizer.state_dict()

        if save_checkpoint:
            checkpoint_dir = _checkpoint_dir(out_dir, iter_num)
            if master_process:
                print(f"saving checkpoint to {checkpoint_dir}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                raw_model.save_pretrained(checkpoint_dir)
                torch.save(optimizer_state_dict, os.path.join(checkpoint_dir, "optimizer.pt"))
                if scaler.is_enabled():
                    torch.save(scaler.state_dict(), os.path.join(checkpoint_dir, "scaler.pt"))
                trainer_state = {
                    "global_step": iter_num,
                    "best_val_loss": best_val_loss,
                    "tokens_trained": tokens_trained,
                }
                with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=2, sort_keys=True)
                    f.write("\n")
            if ddp:
                dist.barrier()
            _save_rng_state(checkpoint_dir)
            if ddp:
                dist.barrier()
            if master_process and val_improved:
                update_best_checkpoint(run_dir=out_dir, checkpoint_dir=checkpoint_dir)
            if master_process and keep_last_checkpoints is not None:
                prune_old_checkpoints(run_dir=out_dir, keep_last=int(keep_last_checkpoints))

        if use_zero1 and master_process and hasattr(optimizer, "_all_state_dicts"):
            optimizer._all_state_dicts = []

        del optimizer_state_dict
        if ddp:
            dist.barrier()

    if eval_only:
        break

    # Stop once we've completed max_iters optimizer updates. We run the eval/checkpoint
    # block at iter_num == max_iters (when aligned with eval_interval) before exiting.
    if iter_num >= max_iters:
        break

    # Forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # In DDP training we only need to sync gradients at the last micro step.
            # The official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # Looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss = model(X, Y)
            # Average gradients across micro-steps so accumulation simulates a larger batch.
            loss = loss / gradient_accumulation_steps
        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        if use_token_stream_dataloader:
            X, Y = _get_train_batch_stream()
        elif data_rng_mode == "stateless":
            train_batch_id += 1
            X, Y = get_batch("train", batch_id=train_batch_id)
        else:
            X, Y = _get_train_batch_stateful()
        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # Clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm.item() > grad_clip:
            clip_time += 1
    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    # Update total tokens trained
    tokens_trained += tokens_per_iter
    tokens_trained_B = tokens_trained / 1e9  # Convert to billions

    if iter_num % log_interval == 0:
        momentum_norm = None
        momentum_norm_sq = None
        momentum_div = None
        if use_zero1:
            acc_device = device if device_type == "cuda" else "cpu"
            momentum_sq = torch.zeros((), device=acc_device, dtype=torch.float64)
            momentum_sq_sq = torch.zeros((), device=acc_device, dtype=torch.float64)
            for state in optimizer.optim.state.values():
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is not None:
                    momentum_sq += exp_avg.detach().float().pow(2).sum().to(dtype=torch.float64)
                if exp_avg_sq is not None:
                    momentum_sq_sq += exp_avg_sq.detach().float().pow(2).sum().to(dtype=torch.float64)
            if backend != "nccl":
                momentum_sq = momentum_sq.cpu()
                momentum_sq_sq = momentum_sq_sq.cpu()
            dist.all_reduce(momentum_sq, op=dist.ReduceOp.SUM)
            dist.all_reduce(momentum_sq_sq, op=dist.ReduceOp.SUM)
            momentum_norm = float(torch.sqrt(momentum_sq).item())
            momentum_norm_sq = float(torch.sqrt(momentum_sq_sq).item())
            momentum_div = momentum_norm / (np.sqrt(momentum_norm_sq) + 1e-8)

        if master_process:
            # Convert back to the unscaled (per-microbatch) loss for logging.
            lossf = loss.item() * gradient_accumulation_steps  # note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            tokens_per_sec = tokens_per_iter / dt
            tokens_per_sec_M = tokens_per_sec / 1_000_000
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
                f"mfu {running_mfu * 100:.2f}%, tps (M) {tokens_per_sec_M:.2f}, "
                f"tokens trained {tokens_trained_B:.2f}B"
            )

            params = [param for _, param in model.named_parameters()]
            total_param_norm = 0.0
            for param in params:
                param_norm = param.data.norm(2)
                total_param_norm += param_norm.item() ** 2
            total_param_norm = total_param_norm**0.5

            if not use_zero1:
                momentum_sq = torch.zeros((), device=device, dtype=torch.float64)
                momentum_sq_sq = torch.zeros((), device=device, dtype=torch.float64)
                for state in optimizer.state.values():
                    exp_avg = state.get("exp_avg")
                    exp_avg_sq = state.get("exp_avg_sq")
                    if exp_avg is not None:
                        momentum_sq += exp_avg.detach().float().pow(2).sum().to(dtype=torch.float64)
                    if exp_avg_sq is not None:
                        momentum_sq_sq += exp_avg_sq.detach().float().pow(2).sum().to(dtype=torch.float64)
                momentum_norm = float(torch.sqrt(momentum_sq).item())
                momentum_norm_sq = float(torch.sqrt(momentum_sq_sq).item())
                momentum_div = momentum_norm / (np.sqrt(momentum_norm_sq) + 1e-8)
            if wandb_log:
                wandb.log(
                    {
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "param_norm": total_param_norm,
                        "momentum_norm": momentum_norm,
                        "momentum_norm_sq": momentum_norm_sq,
                        "momentum_div": momentum_div,
                        "train/clip_rate": clip_time / (iter_num + 1),
                        "train/grad_norm": total_norm.item() if grad_clip != 0.0 else 0.0,
                        "train/iter_time_ms": dt * 1000,
                        "train/mfu": running_mfu * 100,
                        "train/tokens_per_sec_M": tokens_per_sec_M,
                        "train/tokens_trained_B": tokens_trained_B,
                        "gpu/memory_allocated_MB": torch.cuda.memory_allocated() / (1024 * 1024),
                        "gpu/max_memory_allocated_MB": torch.cuda.max_memory_allocated() / (1024 * 1024),
                    },
                    step=iter_num,
                )
                # GDN muP analysis: per-group weight norms, gradient norms, and update magnitudes.
                if _is_gdn_model:
                    _log_gdn_group_stats(raw_model, optimizer, iter_num, lr)
    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()
