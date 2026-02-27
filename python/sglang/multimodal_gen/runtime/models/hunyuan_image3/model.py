"""
HunyuanImage-3.0 TP/EP Transformer backbone.

Drop-in replacement for the official HunyuanImage3Model transformer,
using multimodal_gen TP linear layers and srt FusedMoE.

The forward interface is compatible with the official model so that it can
be used as `model.model` inside `HunyuanImage3ForCausalMM`.
"""

import logging
import re
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

from sglang.multimodal_gen.runtime.distributed import (
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q, k, cos, sin):
    # cos/sin are [batch, seq_len, head_dim]; unsqueeze at dim=1 for head broadcast
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class HunyuanRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# TP Attention - interface-compatible with official HunyuanImage3SDPAAttention
# ---------------------------------------------------------------------------
class TPHunyuanImage3Attention(nn.Module):
    """
    TP-sharded attention that is interface-compatible with the official
    HunyuanImage3SDPAAttention. Uses QKVParallelLinear + RowParallelLinear.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = "self"

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.use_rotary_pos_emb = getattr(config, "use_rotary_pos_emb", True)

        tp_size = get_tp_world_size()
        self.num_heads_per_tp = self.num_heads // tp_size
        self.num_kv_heads_per_tp = max(1, self.num_key_value_heads // tp_size)
        self.num_kv_groups_per_tp = self.num_heads_per_tp // self.num_kv_heads_per_tp
        self.q_size = self.num_heads_per_tp * self.head_dim
        self.kv_size = self.num_kv_heads_per_tp * self.head_dim

        bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=bias,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=bias,
        )

        if self.use_qk_norm:
            self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = False,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(bsz, q_len, self.num_heads_per_tp, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads_per_tp, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads_per_tp, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb and custom_pos_emb is not None:
            cos, sin = custom_pos_emb
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_qk_norm:
            q = self.query_layernorm(q)
            k = self.key_layernorm(k)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            q = q.to(k.dtype)

        k = _repeat_kv(k, self.num_kv_groups_per_tp)
        v = _repeat_kv(v, self.num_kv_groups_per_tp)

        if q.device.type == "cuda" and attention_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output, _ = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ---------------------------------------------------------------------------
# TP MLP
# ---------------------------------------------------------------------------
class TPHunyuanMLP(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = (
            config.intermediate_size
            if isinstance(config.intermediate_size, int)
            else config.intermediate_size[layer_idx]
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# TP MoE using srt's FusedMoE
# ---------------------------------------------------------------------------
class TPHunyuanMoE(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.moe.topk import TopK

        self.tp_size = get_tp_world_size()

        if isinstance(config.moe_topk, list):
            top_k = config.moe_topk[layer_idx]
        else:
            top_k = config.moe_topk

        intermediate_size = config.intermediate_size
        if config.moe_intermediate_size is not None:
            intermediate_size = (
                config.moe_intermediate_size
                if isinstance(config.moe_intermediate_size, int)
                else config.moe_intermediate_size[layer_idx]
            )

        self.topk = TopK(
            top_k=top_k,
            layer_id=layer_idx,
            renormalize=top_k > 1,
        )

        self.experts = FusedMoE(
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            reduce_results=False,
            layer_id=layer_idx,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size, config.num_experts, bias=False
        )

        self.shared_mlp = None
        if getattr(config, "use_mixed_mlp_moe", 0) > 0:
            if isinstance(config.num_shared_expert, list):
                num_shared_expert = config.num_shared_expert[layer_idx]
            else:
                num_shared_expert = config.num_shared_expert
            self.shared_mlp = TPHunyuanMLP.__new__(TPHunyuanMLP)
            nn.Module.__init__(self.shared_mlp)
            self.shared_mlp.gate_up_proj = MergedColumnParallelLinear(
                input_size=config.hidden_size,
                output_sizes=[config.intermediate_size * num_shared_expert] * 2,
                bias=False,
            )
            self.shared_mlp.down_proj = RowParallelLinear(
                input_size=config.intermediate_size * num_shared_expert,
                output_size=config.hidden_size,
                bias=False,
                reduce_results=False,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_mlp is not None:
            shared_output = self.shared_mlp(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


# ---------------------------------------------------------------------------
# TP Decoder Layer - interface-compatible with official
# ---------------------------------------------------------------------------
def _is_moe(config) -> bool:
    num_experts = getattr(config, "num_experts", None)
    if isinstance(num_experts, int):
        return num_experts > 1
    if isinstance(num_experts, list) and num_experts:
        return max(num_experts) > 1
    return False


class TPHunyuanImage3DecoderLayer(nn.Module):
    """Interface-compatible with the official HunyuanImage3DecoderLayer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = TPHunyuanImage3Attention(config, layer_idx)

        moe_layer_start = getattr(config, "moe_layer_num_skipped", 0)
        if _is_moe(config) and layer_idx >= moe_layer_start:
            self.mlp = TPHunyuanMoE(config, layer_idx)
        else:
            self.mlp = TPHunyuanMLP(config, layer_idx)

        self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunyuanRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# TP Transformer backbone - drop-in for official HunyuanImage3Model
# ---------------------------------------------------------------------------
class TPHunyuanImage3Model(nn.Module):
    """
    TP-sharded transformer backbone with the same interface as the official
    HunyuanImage3Model. Attributes: wte, layers, ln_f.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.add_classification_head = getattr(config, "add_classification_head", False)

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [
                TPHunyuanImage3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

        if not self.add_classification_head:
            self.ln_f = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        mode: str = "gen_text",
        first_step: Optional[bool] = None,
        post_token_len: int = None,
        num_image_tokens: int = None,
        gen_timestep_scatter_index: Optional[torch.Tensor] = None,
        num_special_tokens: int = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                mode=mode,
                first_step=first_step,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# Weight loading utility
# ---------------------------------------------------------------------------
_EXPERT_RE = re.compile(
    r"(layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_and_up_proj|down_proj)\.weight"
)


def _unpack_qkv(qkv_weight: torch.Tensor, config) -> Tuple[torch.Tensor, ...]:
    """Split the packed QKV weight into separate Q, K, V tensors.

    Official checkpoint stores QKV interleaved as
    (num_kv_heads, num_kv_groups + 2, head_dim, hidden_size).
    """
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.attention_head_dim
    hidden_size = config.hidden_size
    num_kv_groups = num_heads // num_kv_heads

    qkv = qkv_weight.reshape(num_kv_heads, num_kv_groups + 2, head_dim, hidden_size)
    q, k, v = torch.split(qkv, (num_kv_groups, 1, 1), dim=1)
    q = q.reshape(-1, hidden_size)
    k = k.reshape(-1, hidden_size)
    v = v.reshape(-1, hidden_size)
    return q, k, v


def load_tp_weights(
    model: nn.Module,
    weights: Iterable[Tuple[str, torch.Tensor]],
    config,
) -> int:
    """Load weights into the TP model.

    Handles:
    - QKV packed weights -> QKVParallelLinear (q/k/v shards)
    - Expert fused gate_and_up_proj -> FusedMoE w13 (w1 + w3 shards)
    - Expert down_proj -> FusedMoE w2
    - Shared MLP gate_and_up_proj -> MergedColumnParallelLinear (0/1 shards)
    - MoE gate wg.weight -> ReplicatedLinear
    - Other weights (layernorms, embedding, etc.) -> direct copy
    """
    params_dict = dict(model.named_parameters())
    loaded_count = 0

    for name, loaded_weight in weights:
        if "rotary_emb" in name:
            continue

        # --- Expert weights (FusedMoE) ---
        m = _EXPERT_RE.match(name)
        if m:
            layer_prefix = m.group(1)
            expert_id = int(m.group(2))
            weight_type = m.group(3)

            if weight_type == "gate_and_up_proj":
                param_name = f"{layer_prefix}.experts.w13_weight"
                param = params_dict.get(param_name)
                if param is None:
                    logger.warning("Missing param %s for expert %d", param_name, expert_id)
                    continue
                half = loaded_weight.shape[0] // 2
                gate_w = loaded_weight[:half]
                up_w = loaded_weight[half:]
                param.weight_loader(param, gate_w, param_name, shard_id="w1", expert_id=expert_id)
                param.weight_loader(param, up_w, param_name, shard_id="w3", expert_id=expert_id)
            else:  # down_proj
                param_name = f"{layer_prefix}.experts.w2_weight"
                param = params_dict.get(param_name)
                if param is None:
                    logger.warning("Missing param %s for expert %d", param_name, expert_id)
                    continue
                param.weight_loader(param, loaded_weight, param_name, shard_id="w2", expert_id=expert_id)

            loaded_count += 1
            continue

        # --- QKV packed weight ---
        if ".self_attn.qkv_proj." in name:
            param = params_dict.get(name)
            if param is not None:
                q, k, v = _unpack_qkv(loaded_weight, config)
                param.weight_loader(param, q, "q")
                param.weight_loader(param, k, "k")
                param.weight_loader(param, v, "v")
                loaded_count += 1
            continue

        # --- Shared MLP gate_and_up_proj -> gate_up_proj ---
        if ".shared_mlp.gate_and_up_proj." in name:
            mapped = name.replace("gate_and_up_proj", "gate_up_proj")
            param = params_dict.get(mapped)
            if param is not None:
                half = loaded_weight.shape[0] // 2
                param.weight_loader(param, loaded_weight[:half], 0)   # gate -> shard 0
                param.weight_loader(param, loaded_weight[half:], 1)   # up   -> shard 1
                loaded_count += 1
            continue

        # --- MoE gate: wg.weight -> weight ---
        if "mlp.gate.wg." in name:
            name = name.replace("wg.", "")

        # --- Direct match ---
        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            loaded_count += 1

    logger.info("Loaded %d weight tensors into TP transformer", loaded_count)
    return loaded_count
