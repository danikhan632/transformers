# coding=utf-8
"""Flax Qwen3 MoE model."""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from ...modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from ...modeling_flax_utils import ACT2FN, FlaxPreTrainedModel
from ...utils import logging
from ..qwen3.modeling_flax_qwen3 import FlaxQwen3Attention, FlaxQwen3RMSNorm
from .configuration_qwen3_moe import Qwen3MoeConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Qwen3MoeConfig"
_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-MoE-15B-A2B"

QWEN3_MOE_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
"""


class FlaxQwen3MoeMLP(nn.Module):
    config: Qwen3MoeConfig
    intermediate_size: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_size
        inner_dim = self.intermediate_size or self.config.intermediate_size
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.gate_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.up_proj = nn.Dense(inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.down_proj = nn.Dense(embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.act = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = self.act(self.gate_proj(hidden_states))
        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class FlaxQwen3MoeSparseMoeBlock(nn.Module):
    config: Qwen3MoeConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.num_experts = self.config.num_experts
        self.top_k = self.config.num_experts_per_tok
        self.norm_topk_prob = self.config.norm_topk_prob
        self.gate = nn.Dense(self.num_experts, use_bias=False, dtype=self.dtype, kernel_init=kernel_init)
        self.experts = [
            FlaxQwen3MoeMLP(
                self.config, intermediate_size=self.config.moe_intermediate_size, dtype=self.dtype, name=f"expert_{i}"
            )
            for i in range(self.num_experts)
        ]

    def __call__(self, hidden_states):
        batch, seq_len, dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, dim)
        router_logits = self.gate(flat_states)
        routing_weights = jax.nn.softmax(router_logits, axis=-1)
        if self.top_k < self.num_experts:
            topk_val, topk_idx = lax.top_k(routing_weights, self.top_k)
            if self.norm_topk_prob:
                topk_val = topk_val / topk_val.sum(axis=-1, keepdims=True)
            routing_weights = (
                jnp.zeros_like(routing_weights).at[jnp.arange(flat_states.shape[0])[:, None], topk_idx].set(topk_val)
            )
        expert_outputs = jnp.stack([expert(flat_states) for expert in self.experts], axis=1)
        flat_output = jnp.einsum("bn,bnd->bd", routing_weights, expert_outputs)
        hidden_states = flat_output.reshape(batch, seq_len, dim)
        return hidden_states, router_logits


class FlaxQwen3MoeDecoderLayer(nn.Module):
    config: Qwen3MoeConfig
    layer_idx: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = FlaxQwen3RMSNorm(self.config, dtype=self.dtype)
        self.self_attn = FlaxQwen3Attention(self.config, self.layer_idx, dtype=self.dtype)
        self.post_attention_layernorm = FlaxQwen3RMSNorm(self.config, dtype=self.dtype)
        if (self.layer_idx not in self.config.mlp_only_layers) and (
            self.config.num_experts > 0 and (self.layer_idx + 1) % self.config.decoder_sparse_step == 0
        ):
            self.mlp = FlaxQwen3MoeSparseMoeBlock(self.config, dtype=self.dtype)
        else:
            self.mlp = FlaxQwen3MoeMLP(self.config, dtype=self.dtype, intermediate_size=self.config.intermediate_size)

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        attn_output = outputs[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, router_logits) + outputs[1:]


class FlaxQwen3MoeLayerCollection(nn.Module):
    config: Qwen3MoeConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.blocks = [
            FlaxQwen3MoeDecoderLayer(self.config, i, dtype=self.dtype, name=str(i))
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            hidden_states, router_logits = layer_outputs[:2]
            if output_router_logits:
                all_router_logits += (router_logits,)
            if output_attentions:
                all_attentions += (layer_outputs[2],)

        outputs = (hidden_states, all_hidden_states, all_attentions, all_router_logits)
        return outputs


class FlaxQwen3MoeModule(nn.Module):
    config: Qwen3MoeConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)
        self.embed_tokens = nn.Embed(
            self.config.vocab_size,
            self.hidden_size,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )
        self.layers = FlaxQwen3MoeLayerCollection(self.config, dtype=self.dtype)
        self.norm = FlaxQwen3RMSNorm(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        input_embeds = self.embed_tokens(input_ids.astype("i4"))
        input_embeds = input_embeds * (self.config.hidden_size**0.5)

        outputs = self.layers(
            input_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[2],
            router_logits=outputs[3],
        )


class FlaxQwen3MoePreTrainedModel(FlaxPreTrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: Qwen3MoeConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        random_params = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)["params"]
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params


class FlaxQwen3MoeModel(FlaxQwen3MoePreTrainedModel):
    module_class = FlaxQwen3MoeModule


class FlaxQwen3MoeForCausalLMModule(nn.Module):
    config: Qwen3MoeConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.model = FlaxQwen3MoeModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_router_logits=output_router_logits,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_kernel = self.model.variables["params"]["embed_tokens"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return FlaxCausalLMOutput(
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class FlaxQwen3MoeForCausalLM(FlaxQwen3MoePreTrainedModel):
    module_class = FlaxQwen3MoeForCausalLMModule

    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        batch_size, seq_length = input_ids.shape
        past_key_values = self.init_cache(batch_size, max_length)
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            position_ids = attention_mask.cumsum(axis=-1) - 1
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))
        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
            "position_ids": position_ids,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs


__all__ = ["FlaxQwen3MoeForCausalLM", "FlaxQwen3MoeModel", "FlaxQwen3MoePreTrainedModel"]
