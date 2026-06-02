"""
Fine-grained parameter groups for GDN + Adam (muP) training.

Per mup_adamw.md, the LR assignment for GDN with muP is:

  | Group                     | lr_mult        | weight_decay |
  |---------------------------|----------------|--------------|
  | Hidden weights (q/k/v/o/g/a/b proj, MLP) | d_base/d | yes |
  | Embedding (wte, tied lm_head)            | embedding_lr_mult | no |
  | A_log, dt_bias (special scalars)         | 1.0 (standard)    | no |
  | RMSNorm / norm weights                   | hidden_lr_mult     | no |

Adam handles per-parameter gradient scale differences automatically via second
moments, so unlike SGD we do NOT need extra multipliers for a_proj/b_proj or
A_log/dt_bias beyond what muP dictates.
"""

from __future__ import annotations
import torch.nn as nn


def get_gdn_adam_param_groups(
    model: nn.Module,
    weight_decay: float,
    *,
    hidden_lr_mult: float = 1.0,
    embedding_lr_mult: float = 1.0,
) -> list[dict]:
    """Build AdamW param groups for a GDN model following the mup_adamw.md spec.

    Args:
        model: The GDN model (may be wrapped in DDP — unwrap before calling).
        weight_decay: Weight decay for hidden weight matrices.
        hidden_lr_mult: muP hidden-layer LR multiplier (d_base / d).
        embedding_lr_mult: muP embedding LR multiplier.
    """
    # Embedding params (wte; lm_head is tied so appears here too)
    embedding_params: set[nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    embedding_params.add(p)

    # Special GDN scalar params: A_log and dt_bias — lr = 1.0 (not reduced).
    # Use attribute-based duck-typing to work with both model.gdn and model.gdn-mup.
    scalar_params: set[nn.Parameter] = set()
    for module in model.modules():
        if hasattr(module, 'A_log') and hasattr(module, 'dt_bias'):
            if hasattr(module, 'A_log') and isinstance(module.A_log, nn.Parameter):
                scalar_params.add(module.A_log)
            if hasattr(module, 'dt_bias') and isinstance(module.dt_bias, nn.Parameter):
                scalar_params.add(module.dt_bias)

    # No-weight-decay params: RMSNorm / norm weights, and param-level _no_weight_decay
    no_wd_params: set[nn.Parameter] = set()
    for module in model.modules():
        if module.__class__.__name__ in {"RMSNorm", "RMSNormLinear", "FusedRMSNormGated"}:
            for p in module.parameters():
                if isinstance(p, nn.Parameter):
                    no_wd_params.add(p)
    for name, param in model.named_parameters():
        if isinstance(param, nn.Parameter) and getattr(param, "_no_weight_decay", False):
            no_wd_params.add(param)

    hidden_decay: list[nn.Parameter] = []
    hidden_no_decay: list[nn.Parameter] = []
    emb_list: list[nn.Parameter] = []
    scalar_list: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in embedding_params:
            emb_list.append(param)
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
    if scalar_list:
        # A_log and dt_bias: lr = 1.0 per mup_adamw.md, no weight decay
        groups.append({"params": scalar_list, "weight_decay": 0.0,
                       "lr_mult": 1.0, "name": "gdn_scalar"})

    return groups
