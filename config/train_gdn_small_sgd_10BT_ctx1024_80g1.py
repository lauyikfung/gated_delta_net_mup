from train_adam_config import TrainAdamConfigBase

_batch_size = 30

# GDN-specific SGD param group LR multipliers.
#
# In GDN, different parameter types have very different gradient norm scales:
#   - Main projections (q/k/v/o/g_proj): gradient norm ∝ sqrt(out_dim * in_dim)
#     e.g. q_proj [key_dim×hidden_size] with key_dim=192, hidden_size=256: ~sqrt(49152)≈222
#   - Gate projections (a_proj, b_proj): output dim = num_v_heads=6,
#     gradient norm ≈ sqrt(6×256)≈39, i.e. ~6× smaller
#   - Scalar params (A_log, dt_bias): 6 elements each,
#     gradient norm ≈ sqrt(6)≈2.5, i.e. ~90× smaller
#
# Additionally, with std≈0 initialization, a_proj(x) ≈ 0 for normalized x,
# making the gate input-independent → near-zero gradients → SGD can't update a_proj.
# The muP-style init in gdn.py's post_init (std=factor/sqrt(hidden_size)=0.02 at d=256)
# fixes this by giving a_proj meaningful initial activations and non-zero gradients.
#
# These multipliers compensate for the remaining gradient-norm disparity so SGD gives
# comparable effective update magnitudes across all GDN parameter groups.
gdn_gate_lr_mult = 1.0    # extra multiplier for a_proj, b_proj, original 6.0
gdn_scalar_lr_mult = 1.0 # extra multiplier for A_log, dt_bias, original 50.0

CONFIG = TrainAdamConfigBase(
    # Wandb configs
    wandb_log=True,
    wandb_project="nanogpt-pro",
    # Tokenizer configs
    tokenizer="gpt2",
    # Model configs
    num_hidden_layers=8,
    num_attention_heads=6,
    hidden_size=512,
    head_dim=-1,
    dropout=0.0,
    bias=False,
    using_groupnorm=False,
    use_qk_rmsnorm=False,
    # muP-style init: hidden params at std=hidden_init_std_factor/sqrt(hidden_size)
    # With hidden_size=256 and factor=0.32: std=0.32/16=0.02, matching non-mup std=0.02
    embedding_init_std=0.02,
    hidden_init_std_factor=0.32,
    # muP — hidden_size_base=256 aligns at d=256:
    #   logits_scale = 256/256 = 1.0  (matches non-mup)
    #   mymup gate_mult = 6*sqrt(256/256) = 6x  (matches non-mymup baseline at d=256)
    #   mymup scalar_mult = 50*sqrt(256/256) = 50x  (matches non-mymup baseline at d=256)
    mup=True,
    mymup=False,
    hidden_size_base=256,
    embedding_lr_multiplier=1.0,
    # Training configs
    batch_size=_batch_size,
    global_batch_size=480,
    block_size=1024,
    gradient_accumulation_steps=480 // _batch_size,
    max_iters=20000,
    lr_decay_iters=20000,
    eval_interval=1000,
    eval_iters=200,
    log_interval=10,
    save_checkpoints=True,
    keep_last_checkpoints=3,
    # Optimizer configs — SGD with per-group LR compensation
    #
    # With hidden_size=256=hidden_size_base, hidden_lr_mult=1.0 (hardcoded in SGD file):
    #   Standard hidden params: lr_base * 1.0
    #   Gate params (a_proj, b_proj): lr_base * 1.0 * 6 = 0.12
    #   Scalar params (A_log, dt_bias): lr_base * 1.0 * 50 = 1.0
    #
    optimizer_name="sgd",
    learning_rate_base=2e-2,
    min_lr_base=5e-5,
    weight_decay=0.0,
    beta1=0.98,   # momentum (0.95 gives smoother updates than 0.9)
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=2000,
    schedule="cosine",
    # System configs
    compile=True,
    model_type="gdn",
)

globals().update(CONFIG.model_dump(exclude_unset=True))
