from train_adam_config import TrainAdamConfigBase

_batch_size = 60

CONFIG = TrainAdamConfigBase(
    # Wandb configs
    wandb_log=True,
    wandb_project="nanogpt-pro",
    # Tokenizer configs
    tokenizer="gpt2",
    # Model configs
    num_hidden_layers=8, # original is 12
    num_attention_heads=6,
    hidden_size=512,
    head_dim=-1,
    dropout=0.0,
    bias=False,
    using_groupnorm=False,
    use_qk_rmsnorm=False,
    # Embedding and hidden init
    embedding_init_std=0.02,
    hidden_init_std_factor=0.5,
    # muP
    mup=True,
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
    # Optimizer configs
    optimizer_name="adamw",
    learning_rate_base=1e-3,
    min_lr_base=5e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    decay_lr=True,
    warmup_iters=2000,
    schedule="cosine",
    # System configs
    compile=True,
    model_type="gpt-mha-rope",
)

globals().update(CONFIG.model_dump(exclude_unset=True))
