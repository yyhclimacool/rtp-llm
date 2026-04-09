import functools
import logging
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models.qwen_v2_moe import Qwen2Moe, QWenV2MoeWeight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    zeros,
)

logger = logging.getLogger(__name__)


def _reconstruct_d2t_from_t2d(ts: List[torch.Tensor]) -> torch.Tensor:
    """Reconstruct the d2t (draft→target) index array from the t2d boolean mask.

    SpecForge checkpoints may save a corrupted d2t where identity-mapped entries
    (the first N positions) are all zero instead of [0, 1, 2, ..., N-1].
    Reconstructing from t2d is always correct because t2d is the ground truth:
        d2t = sorted indices where t2d is True
    """
    t2d = ts[0]
    d2t = torch.where(t2d.bool())[0]
    logger.info(
        "Reconstructed d2t from t2d: %d draft tokens mapped from %d target vocab",
        d2t.shape[0],
        t2d.shape[0],
    )
    return d2t


class QWenV3MoeWeight(QWenV2MoeWeight):
    def __init__(
        self,
        model_config,
        parallelism_config,
        hw_kernel_config,
        kv_cache_config,
        merge_lora=False,
        vit_config=None,
        prefix="",
        **kwargs: Any,
    ):
        super().__init__(
            model_config=model_config,
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
            kv_cache_config=kv_cache_config,
            merge_lora=merge_lora,
            vit_config=vit_config,
            prefix=prefix,
            **kwargs,
        )
        self.bias = False

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            routed_scaling_factor=1.0,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ]
                        + [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            )
                        ],
                        stack_moe_w1,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]


class Qwen3Moe(Qwen2Moe):
    @staticmethod
    def get_weight_cls():
        return QWenV3MoeWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.qk_norm = True
        config.moe_style = 1
        return config

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel

        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size

        self.py_model = GenericMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class Qwen3MoeEagle3Weight(QWenV2Weight):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = False
        self._use_qk_norm = True

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
        ]
        assert self._num_layers == 1
        for layer in range(self._num_layers):
            layer_weights_tmp = self._get_hf_layer_weight_info(layer)
            # Eagle3 embeddingPost already applies norms before concatenation.
            # Remove pre_ln_gamma to avoid shape mismatch (gamma [H] vs hidden [2H]).
            layer_weights_tmp = [
                w
                for w in layer_weights_tmp
                if getattr(w, "name", None) != W.pre_ln_gamma
            ]
            layer_weights_tmp.extend(
                [
                    AtomicWeight(
                        W.eagle3_fc_proj,
                        [CkptWeightInfo("fc.weight", identity)],
                        transpose,
                    ),
                    AtomicWeight(
                        W.eagle3_fc_norm_gamma,
                        [CkptWeightInfo("model.layers.0.hidden_norm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.eagle3_input_norm_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.0.input_layernorm.weight", identity
                            )
                        ],
                        identity,
                    ),
                ]
            )
            layer_weights.append(layer_weights_tmp)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Qwen3MoeEagle3(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3MoeEagle3Weight


register_model("qwen_3_moe", Qwen3Moe, ["Qwen3MoeForCausalLM"])
register_model("qwen_3_moe_eagle3", Qwen3MoeEagle3, ["Qwen3MoeForCausalLMEagle"])
register_model("qwen3_coder_moe", Qwen3Moe, [])


# ---------------------------------------------------------------------------
# Specforge LlamaForCausalLMEagle3 draft-model support
# ---------------------------------------------------------------------------
#
# Specforge trains Eagle3 draft models whose checkpoint layout differs from
# the official Qwen3-MoE Eagle3 format:
#
#   fc.weight                                [hidden, 3*hidden]    eagle3_fc_proj
#   lm_head.weight                           [draft_vocab, hidden] lm_head
#   norm.weight                              [hidden]              final_ln_gamma
#   midlayer.self_attn.q_proj.weight         [Q_dim, 2*hidden]     ↘
#   midlayer.self_attn.k_proj.weight         [K_dim, 2*hidden]      attn_qkv_w
#   midlayer.self_attn.v_proj.weight         [V_dim, 2*hidden]     ↗
#   midlayer.self_attn.o_proj.weight         [hidden, Q_dim]       attn_o_w
#   midlayer.mlp.gate_proj.weight            [inter, hidden]       ffn_w1
#   midlayer.mlp.up_proj.weight              [inter, hidden]       ffn_w3
#   midlayer.mlp.down_proj.weight            [hidden, inter]       ffn_w2
#   midlayer.hidden_norm.weight              [hidden]              eagle3_fc_norm_gamma
#   midlayer.input_layernorm.weight          [hidden]              pre_ln_gamma +
#                                                                  eagle3_input_norm_gamma
#   midlayer.post_attention_layernorm.weight [hidden]              post_ln_gamma
#   d2t / t2d                                                      (ignored)
#
# IMPORTANT – embed_tokens:
#   Specforge intentionally omits embed_tokens.weight from the saved checkpoint
#   (it expects the serving framework to reuse the main model's embedding table).
#   Before starting the server, merge the main model's embedding into the eagle3
#   checkpoint once using the provided merge_embed_tokens.py helper script.
# ---------------------------------------------------------------------------


class SpecforgeLlamaEagle3Weight(QWenV2Weight):
    """Weight loader for specforge LlamaForCausalLMEagle3 draft checkpoints."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bias = False
        self._use_qk_norm = False  # Llama-style attention: no qk_norm

    def _get_weight_info(self):
        assert self._num_layers == 1, (
            f"SpecforgeLlamaEagle3 draft model must have exactly 1 transformer "
            f"layer (num_hidden_layers=1), got {self._num_layers}"
        )

        layer_weights = [self._get_specforge_layer_weight_info()]

        weights = [
            # embed_tokens.weight must be present in the checkpoint.
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            # lm_head maps to draft_vocab_size rows; loaded as-is.
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
                identity,
            ),
            # Final RMSNorm (self.norm) applied by compute_logits before lm_head.
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            # Draft-to-target token ID mapping (1-D int32 lookup table).
            # d2t[draft_token] -> target_token  shape: [draft_vocab_size]
            # Reconstructed from the t2d boolean mask to avoid corrupted d2t
            # values in SpecForge checkpoints (identity-mapped positions saved as
            # all-zero).  data_type=torch.int32 preserves exact integer values.
            AtomicWeight(
                W.eagle3_d2t,
                [CkptWeightInfo("t2d", identity)],
                _reconstruct_d2t_from_t2d,
                data_type=torch.int32,
            ),
            # Target-to-draft boolean mask (t2d[target_token] = True if in draft vocab).
            # Not used at runtime; kept for checkpoint compatibility.
            AtomicWeight(
                W.eagle3_t2d,
                [CkptWeightInfo("t2d", identity)],
                identity,
            ),
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_specforge_layer_weight_info(self) -> List[WeightModule]:
        """Build weight descriptors for the single 'midlayer' transformer block.

        Eagle3 embeddingPost already applies hidden_norm and input_layernorm
        before concatenation, so the standard pre-attention layernorm slot
        (pre_ln_gamma) is intentionally NOT loaded here.  Loading it would
        cause a shape mismatch: gamma [hidden_size] applied to the
        concatenated hidden [2*hidden_size] inside forwardAttentionBlock.
        """
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
            is_moe=False,
        )
        align_size = self._align_size

        input_ln_key = "midlayer.input_layernorm.weight"

        return [
            # ── Attention ─────────────────────────────────────────────────
            # q/k/v input dimension is 2*hidden (cat of normed_emb + normed_fc).
            # No qk_norm (Llama-style).
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo("midlayer.self_attn.q_proj.weight", identity),
                    CkptWeightInfo("midlayer.self_attn.k_proj.weight", identity),
                    CkptWeightInfo("midlayer.self_attn.v_proj.weight", identity),
                ],
                functools.partial(merge_qkv_hf),
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo("midlayer.self_attn.o_proj.weight", identity)],
                transpose,
                config=attn_config,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo("midlayer.post_attention_layernorm.weight", identity)],
                identity,
                config=attn_config,
            ),
            # ── FFN (SiLU-gated, matching LlamaMLP) ───────────────────────
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [CkptWeightInfo("midlayer.mlp.gate_proj.weight", identity)],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [CkptWeightInfo("midlayer.mlp.up_proj.weight", identity)],
                        functools.partial(transpose_pad, align_size=align_size, dim=0),
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [CkptWeightInfo("midlayer.mlp.down_proj.weight", identity)],
                        functools.partial(transpose_pad, align_size=align_size, dim=1),
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            # ── Eagle3-specific weights ────────────────────────────────────
            # fc: project 3 concatenated main-model hidden layers → hidden_size.
            AtomicWeight(
                W.eagle3_fc_proj,
                [CkptWeightInfo("fc.weight", identity)],
                transpose,
            ),
            # Norm applied to the fc-projected hidden state in embeddingPost.
            AtomicWeight(
                W.eagle3_fc_norm_gamma,
                [CkptWeightInfo("midlayer.hidden_norm.weight", identity)],
                identity,
            ),
            # Norm applied to the token embedding in embeddingPost.
            AtomicWeight(
                W.eagle3_input_norm_gamma,
                [CkptWeightInfo(input_ln_key, identity)],
                identity,
            ),
        ]


class SpecforgeLlamaEagle3(QWenV2):
    """Eagle3 draft model trained by specforge on a Llama/Qwen3-dense backbone."""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        # Llama-style attention: no per-head QK normalisation.
        config.qk_norm = False
        return config

    @staticmethod
    def get_weight_cls():
        return SpecforgeLlamaEagle3Weight


register_model(
    "specforge_llama_eagle3", SpecforgeLlamaEagle3, ["LlamaForCausalLMEagle3"]
)
