from __future__ import annotations

from typing import TypedDict

import torch.nn as nn


class OptimizerParamGroup(TypedDict):
    params: list[nn.Parameter]
    weight_decay: float
    lr_mult: float


def _get_rmsnorm_params(model: nn.Module) -> set[nn.Parameter]:
    rmsnorm_params: set[nn.Parameter] = set()
    for module in model.modules():
        if module.__class__.__name__ in {"RMSNorm", "RMSNormLinear"}:
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            if isinstance(weight, nn.Parameter):
                rmsnorm_params.add(weight)
            if isinstance(bias, nn.Parameter):
                rmsnorm_params.add(bias)
    return rmsnorm_params

def _get_no_wd_params(model: nn.Module) -> set[nn.Parameter]:
    no_wd_params: set[nn.Parameter] = set()
    for module in model.modules():
        if hasattr(module, "_no_weight_decay") and getattr(module, "_no_weight_decay", False):
            for param in module.parameters():
                if isinstance(param, nn.Parameter):
                    no_wd_params.add(param)
    # Also catch parameter-level _no_weight_decay (e.g. A_log, dt_bias in GDN)
    for _, param in model.named_parameters():
        if isinstance(param, nn.Parameter) and getattr(param, "_no_weight_decay", False):
            no_wd_params.add(param)
    return no_wd_params


def _get_embedding_params(model: nn.Module) -> set[nn.Parameter]:
    embedding_params: set[nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            weight = module.weight
            if isinstance(weight, nn.Parameter):
                embedding_params.add(weight)
    return embedding_params


def get_optimizer_param_groups(
    model: nn.Module,
    weight_decay: float,
    *,
    split_embeddings: bool = False,
    embedding_lr_mult: float = 1.0,
    hidden_lr_mult: float = 1.0,
) -> list[OptimizerParamGroup]:
    """Build AdamW param groups.

    When split_embeddings=True, parameters are split into embedding vs hidden groups and each group
    includes an `lr_mult` field for external LR scheduling.
    """
    embedding_lr_mult = float(embedding_lr_mult)
    hidden_lr_mult = float(hidden_lr_mult)

    rmsnorm_params = _get_rmsnorm_params(model)
    no_wd_params = _get_no_wd_params(model)
    rmsnorm_params.update(no_wd_params)
    if not split_embeddings:
        decay_params = []
        no_decay_params = []
        for param in model.parameters():
            if param in rmsnorm_params:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups: list[OptimizerParamGroup] = [
            {"params": decay_params, "weight_decay": weight_decay, "lr_mult": 1.0}
        ]
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr_mult": 1.0})
        return param_groups

    embedding_params = _get_embedding_params(model)

    hidden_decay_params = []
    hidden_no_decay_params = []
    embedding_decay_params = []
    embedding_no_decay_params = []
    for param in model.parameters():
        is_no_decay = param in rmsnorm_params
        is_embedding = param in embedding_params
        if is_embedding:
            if is_no_decay:
                embedding_no_decay_params.append(param)
            else:
                embedding_decay_params.append(param)
        else:
            if is_no_decay:
                hidden_no_decay_params.append(param)
            else:
                hidden_decay_params.append(param)

    param_groups: list[OptimizerParamGroup] = []
    if hidden_decay_params:
        param_groups.append({"params": hidden_decay_params, "weight_decay": weight_decay, "lr_mult": hidden_lr_mult})
    if hidden_no_decay_params:
        param_groups.append({"params": hidden_no_decay_params, "weight_decay": 0.0, "lr_mult": hidden_lr_mult})
    if embedding_decay_params:
        param_groups.append(
            {
                "params": embedding_decay_params,
                "weight_decay": weight_decay,
                "lr_mult": embedding_lr_mult,
            }
        )
    if embedding_no_decay_params:
        param_groups.append({"params": embedding_no_decay_params, "weight_decay": 0.0, "lr_mult": embedding_lr_mult})
    return param_groups
