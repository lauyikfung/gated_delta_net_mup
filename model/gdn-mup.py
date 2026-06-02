from __future__ import annotations
# Copyright from https://github.com/fla-org/flash-linear-attention by Songlin Yang, Yu Zhang
import math
import warnings
from typing import TYPE_CHECKING, Optional, Any

import torch
import torch.nn as nn

import torch.nn.functional as F
from dataclasses import dataclass
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .gpt_base import PastKeyValue
from .rmsnorm import RMSNorm
from .pydantic_config import validate_pretrained_config_kwargs

from transformers.utils import logging
from fla.models.utils import Cache
from fla.layers.attn import Attention
from fla.modules import RMSNorm
from fla.modules import GatedMLP as GatedDeltaNetMLP

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)

from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule



@torch.compile
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        mup: bool = False,
        hidden_size_base: int | None = None,
        **kwargs,
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.output_multiplier = (
            float(hidden_size_base if hidden_size_base is not None else hidden_size) / float(hidden_size)
            if mup
            else 1.0
        )

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = 'fused_recurrent' if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_k_dim), (q, k))
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        if self.output_multiplier != 1.0:
            o = o * self.output_multiplier
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


class Block(GradientCheckpointingLayer):

    def __init__(self, config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_attention_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                qkv_bias=config.attn['qkv_bias'],
                window_size=config.attn['window_size'],
                rope_theta=config.attn['rope_theta'],
                max_position_embeddings=config.block_size,
                layer_idx=layer_idx,
            )
        else:
            self.attn = GatedDeltaNet(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_attention_heads,
                num_v_heads=config.num_v_heads,
                use_gate=config.use_gate,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                mup=config.mup,
                hidden_size_base=config.hidden_size_base,
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedDeltaNetMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values)

        return outputs

@dataclass
class GPTConfig(PretrainedConfig):
    attn_mode: str = "chunk"
    expand_v: float = 2.0
    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    num_v_heads: int | None = None
    hidden_ratio: int | None = 4
    intermediate_size: int | None = None
    hidden_act: str = "swish"
    norm_eps: float = 1e-6
    attn: dict | None = None
    linear_attn_layers: list[int] = None
    
    use_cache: bool = True
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    fuse_norm: bool = True
    fuse_swiglu: bool = True
    # muP knobs (forward/logits scaling + optimizer LR grouping)
    mup: bool = False
    mymup: bool = False
    # Reference width for muP LR scaling (hidden_lr_mult = hidden_size_base / hidden_size)
    hidden_size_base: int = 1024
    # Embedding init std (normal init for tied token embedding / LM head)
    embedding_init_std: float = 0.02
    # Factor for hidden (>=2D) param init; actual std = factor / sqrt(hidden_size)
    hidden_init_std_factor: float = 0.5
    embedding_lr_multiplier: float = 1.0
    model_type = "nanogpt-pro"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6  # head dim 128 suggested by @Grad62304977
    hidden_size: int = 768
    head_dim: int = 96  # Dimension per head for linear attention, please set to hidden_size * 0.75 / num_attention_heads
    block_size: int = 1024  # Maximum sequence length
    bias: bool = False  # Use bias in all linear layers

    def __init__(self, **kwargs: Any) -> None:
        raw = dict(kwargs)
        super().__init__(**validate_pretrained_config_kwargs(type(self), raw))
        if self.num_attention_heads == -1:
            self.num_attention_heads = int(self.hidden_size * 3 // (self.head_dim * 4))
        if self.head_dim == -1:
            self.head_dim = int(self.hidden_size * 0.75 // self.num_attention_heads)
        self.attn = {"num_kv_heads": self.num_attention_heads, "qkv_bias": self.bias, "window_size": None, "rope_theta": 10000,
                     "num_attention_heads": self.num_attention_heads, "layers": self.linear_attn_layers if self.linear_attn_layers is not None else []}  # Default to full attention
        
class GPT(PreTrainedModel):

    config_class = GPTConfig
    base_model_prefix = "nanogpt-pro"
    supports_gradient_checkpointing = True
    _no_split_modules = ['Block']
    _supports_cache_class = True

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_size),
                h=nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
            )
        )
        self.ln_f = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.mup = config.mup
        self.config = config
        self.post_init()
        if config.tie_word_embeddings:
            with torch.no_grad():
                self.lm_head.weight.normal_(
                    mean=0.0,
                    std=float(getattr(config, "embedding_init_std", getattr(config, "initializer_range", 0.02))),
                )
            self.tie_weights()

    def post_init(self):
        # Delegate all SP-vs-muP initialization choices to _init_weights.
        # This avoids a manual initialization pass being overwritten by
        # PreTrainedModel.post_init().
        super().post_init()

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs) -> None:
        self.transformer.wte.weight = self.lm_head.weight
                
    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        *,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor | None, torch.Tensor | None]:
        del position_ids, cache_position, kwargs

        if (idx is None) == (input_ids is None):
            raise ValueError("Exactly one of `idx` or `input_ids` must be provided.")
        if idx is None:
            idx = input_ids

        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache_flag = bool(use_cache) if use_cache is not None else False
        return_dict_flag = bool(return_dict) if return_dict is not None else False
        output_hidden_states_flag = bool(output_hidden_states) if output_hidden_states is not None else False

        if attention_mask is not None and bool(attention_mask.to(dtype=torch.bool).all().item()):
            attention_mask = None

        x = self.transformer.wte(idx)
        hidden_states: tuple[torch.Tensor, ...] | None = (x,) if output_hidden_states_flag else None

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = Cache.from_legacy_cache(past_key_values)
            
        past_key_values: list[PastKeyValue] | None = None
        
        for layer_idx, block in enumerate(self.transformer.h):
            if use_cache_flag or past_key_values is not None or attention_mask is not None:
                x, _, past_key_values = block(
                    x,
                    past_key_value=past_key_values,
                    use_cache=use_cache_flag,
                    attention_mask=attention_mask,
                )
            else:
                x, _, _ = block(x)

            if output_hidden_states_flag:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        x = self.ln_f(x)

        logits_scale = 1.0
        if getattr(self.config, "mup", False):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        if targets is not None:
            logits = self.lm_head(x).float() * logits_scale
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict_flag:
                logits = self.lm_head(x) * logits_scale
            else:
                logits = self.lm_head(x[:, [-1], :]).float() * logits_scale

        if not return_logits:
            logits = None
        if not return_dict_flag:
            return logits, loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )

        
    def _init_weights(self, module: nn.Module):
        if getattr(self.config, "mup", False):
            hidden_std = float(getattr(self.config, "hidden_init_std_factor", 0.5)) / math.sqrt(
                float(self.config.hidden_size)
            )
            embedding_std = float(getattr(self.config, "embedding_init_std", 0.02))
        else:
            hidden_std = float(getattr(self.config, "initializer_range", 0.02))
            embedding_std = hidden_std

        if isinstance(module, GatedDeltaNet) and next(module.parameters()).device.type != 'meta':
            with torch.no_grad():
                module.A_log.copy_(nn.init.uniform_(module.A_log, a=0, b=16).log())
                module.A_log._no_weight_decay = True
                dt = torch.exp(
                    nn.init.uniform_(module.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001),
                ).clamp(min=1e-4)
                # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
                inv_dt = dt + torch.log(-torch.expm1(-dt))
                module.dt_bias.copy_(inv_dt)
                module.dt_bias._no_weight_decay = True
        elif isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=hidden_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=embedding_std)
        elif hasattr(module, 'reset_parameters'):
            module.reset_parameters()


    def crop_block_size(self, block_size):
        block_size_int = int(block_size)
        if block_size_int <= 0:
            raise ValueError(f"block_size must be a positive integer, got {block_size_int}.")

        current = getattr(self.config, "block_size", None)
        if isinstance(current, int):
            current_int = int(current)
            if block_size_int > current_int:
                raise ValueError(f"block_size must be <= {current_int} to crop, got {block_size_int}.")

        setattr(self.config, "block_size", block_size_int)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = (
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.hidden_size // cfg.num_attention_heads,
            cfg.block_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        # return n_params
        return n_params

    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        super().save_pretrained(save_directory, safe_serialization=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args: Any, **kwargs: Any) -> Any:
        config = kwargs.pop("config", None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        if isinstance(model, GPT) and config.tie_word_embeddings:
            model.tie_weights()
        return model
