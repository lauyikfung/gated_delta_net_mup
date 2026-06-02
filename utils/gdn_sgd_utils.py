"""
Fine-grained parameter groups for GDN + SGD training.

SGD doesn't adapt per-parameter learning rates like Adam. In GDN, different
parameter types have very different gradient norm scales:

  - Main projections (q/k/v/o/g_proj): gradient norm ∝ sqrt(out_dim * in_dim)
    e.g. q_proj [192×256]: ~222
  - Gate projections (a_proj, b_proj): output dim = num_v_heads (e.g. 6),
    gradient norm ~39 — ~6x smaller than main projections
  - Scalar params (A_log, dt_bias): dim = num_v_heads (6 elements),
    gradient norm ~2.5 — ~90x smaller than main projections

Adam handles this automatically via per-parameter second-moment scaling.
For SGD, we compensate by assigning higher LR multipliers to the smaller-gradient
parameter groups so all groups receive comparable effective update magnitudes.
"""

from __future__ import annotations
import torch.nn as nn


def get_gdn_sgd_param_groups(
    model: nn.Module,
    weight_decay: float,
    *,
    hidden_lr_mult: float = 1.0,
    embedding_lr_mult: float = 1.0,
    gate_lr_mult: float = 6.0,
    scalar_lr_mult: float = 50.0,
) -> list[dict]:
    """
    Build SGD param groups for a GDN model with per-group LR compensation.

    Args:
        model: The GDN model.
        weight_decay: Weight decay (typically 0 for SGD on GDN).
        hidden_lr_mult: muP hidden-layer LR multiplier (base/width).
        embedding_lr_mult: muP embedding LR multiplier.
        gate_lr_mult: Extra LR multiplier for a_proj/b_proj on top of hidden_lr_mult.
            Compensates for their ~6x smaller gradient norm relative to main projections.
        scalar_lr_mult: Extra LR multiplier for A_log/dt_bias on top of hidden_lr_mult.
            Compensates for their ~90x smaller gradient norm relative to main projections.
    """
    try:
        from model.gdn import GatedDeltaNet
    except ImportError:
        from model.gdn_mup import GatedDeltaNet  # type: ignore[no-redef]

    # Identify embedding params
    embedding_params: set[nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    embedding_params.add(p)

    # Identify GDN-specific params
    gate_params: set[nn.Parameter] = set()   # a_proj, b_proj
    scalar_params: set[nn.Parameter] = set() # A_log, dt_bias

    for module in model.modules():
        if isinstance(module, GatedDeltaNet):
            for p in module.a_proj.parameters():
                gate_params.add(p)
            for p in module.b_proj.parameters():
                gate_params.add(p)
            if hasattr(module, 'A_log') and isinstance(module.A_log, nn.Parameter):
                scalar_params.add(module.A_log)
            if hasattr(module, 'dt_bias') and isinstance(module.dt_bias, nn.Parameter):
                scalar_params.add(module.dt_bias)

    # Identify no-weight-decay params (RMSNorm weights, biases)
    no_wd_params: set[nn.Parameter] = set()
    for module in model.modules():
        if module.__class__.__name__ in {"RMSNorm", "RMSNormLinear", "FusedRMSNormGated"}:
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    no_wd_params.add(p)

    hidden_decay: list[nn.Parameter] = []
    hidden_no_decay: list[nn.Parameter] = []
    emb_list: list[nn.Parameter] = []
    gate_list: list[nn.Parameter] = []
    scalar_list: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in embedding_params:
            emb_list.append(param)
        elif param in gate_params:
            gate_list.append(param)
        elif param in scalar_params:
            scalar_list.append(param)
        elif param in no_wd_params or name.endswith("bias"):
            hidden_no_decay.append(param)
        else:
            hidden_decay.append(param)

    groups: list[dict] = []
    if hidden_decay:
        groups.append({"params": hidden_decay, "weight_decay": weight_decay,
                       "lr_mult": hidden_lr_mult, "name": "hidden_decay"})
    if hidden_no_decay:
        groups.append({"params": hidden_no_decay, "weight_decay": 0.0,
                       "lr_mult": hidden_lr_mult, "name": "hidden_no_decay"})
    if emb_list:
        groups.append({"params": emb_list, "weight_decay": 0.0,
                       "lr_mult": embedding_lr_mult, "name": "embedding"})
    if gate_list:
        groups.append({"params": gate_list, "weight_decay": 0.0,
                       "lr_mult": hidden_lr_mult * gate_lr_mult, "name": "gdn_gate"})
    if scalar_list:
        groups.append({"params": scalar_list, "weight_decay": 0.0,
                       "lr_mult": hidden_lr_mult * scalar_lr_mult, "name": "gdn_scalar"})

    return groups
