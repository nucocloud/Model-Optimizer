# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM sparse attention plugin — registers _SparseVLLMAttention.

Wraps ``vllm.attention.Attention`` to replace the attention computation with
the ModelOpt Triton kernel while reusing vLLM's paged KV cache write path.
"""

import torch
import vllm.attention as vllm_attention

from modelopt.torch.kernels.triton_fa import attention as triton_attention

from ..sparse_attention import SparseAttentionModule, SparseAttentionRegistry


@SparseAttentionRegistry.register({vllm_attention.Attention: "vllm_Attention"})
class _SparseVLLMAttention(SparseAttentionModule):
    """Sparse attention wrapper for vLLM's Attention layer.

    Overrides ``forward()`` directly (same pattern as ``_QuantVLLMAttention``
    in ``quantization/plugins/vllm.py``) to bypass ``SparseAttentionModule``'s
    context-manager machinery and call the Triton kernel directly.
    """

    def _setup(self):
        """Initialize sparse config from method instance."""
        super()._setup()
        method = self._sparse_method_instance
        self._sparsity_n = getattr(method, "sparsity_n", 0)
        self._sparsity_m = getattr(method, "sparsity_m", 4)
        self._num_sink_blocks = getattr(method, "num_sink_blocks", 0)
        self._dense_window_blocks = getattr(method, "dense_window_blocks", 1)
        self._skip_softmax_threshold = getattr(method, "skip_softmax_threshold", None)

    def forward(self, query, key, value, kv_cache, attn_metadata, **kwargs):
        """Forward with sparse attention via Triton kernel.

        When disabled, falls through to normal attention.
        """
        if not self.is_enabled:
            return super().forward(query, key, value, kv_cache, attn_metadata, **kwargs)

        # Step 1: Write K/V to paged cache
        from vllm._custom_ops import reshape_and_cache_flash

        reshape_and_cache_flash(
            key,
            value,
            kv_cache,
            attn_metadata.slot_mapping,
            self.impl.kv_cache_dtype,
            getattr(self.impl, "k_scale", 1.0),
            getattr(self.impl, "v_scale", 1.0),
        )

        # Step 2: Unpack paged KV cache
        k_cache = kv_cache[:, 0]  # [num_blocks, page_size, num_kv_heads, head_dim]
        v_cache = kv_cache[:, 1]
        page_size = k_cache.shape[1]

        output = torch.empty_like(query)
        sm_scale = self.impl.scale

        # Build sparse kernel kwargs
        sparse_kw = {}
        if self._sparsity_n > 0:
            sparse_kw["sparsity_n"] = self._sparsity_n
            sparse_kw["sparsity_m"] = self._sparsity_m
            sparse_kw["num_sink_blocks"] = self._num_sink_blocks
            sparse_kw["dense_window_blocks"] = self._dense_window_blocks
        if self._skip_softmax_threshold:
            sparse_kw["skip_softmax_threshold"] = self._skip_softmax_threshold

        # Paged KV kwargs (shared between prefill and decode)
        paged_kw = {
            "k_cache": k_cache,
            "v_cache": v_cache,
            "page_size": page_size,
        }

        # Step 3: Prefill
        if attn_metadata.num_prefill_tokens > 0:
            pm = attn_metadata.prefill
            n = attn_metadata.num_prefill_tokens
            output[:n] = triton_attention(
                q=query[:n],
                k=query[:0],  # dummy, not used in paged mode
                v=query[:0],
                b_start_loc=pm.query_start_loc,
                b_seq_len=pm.seq_lens_q,
                max_input_len=int(pm.seq_lens_q.max().item()),
                is_causal=True,
                softmax_scale=sm_scale,
                b_seq_len_k=pm.seq_lens,
                max_input_len_k=int(pm.seq_lens.max().item()),
                block_table=pm.block_tables,
                **paged_kw,
                **sparse_kw,
            )

        # Step 4: Decode
        if attn_metadata.num_decode_tokens > 0:
            dm = attn_metadata.decode
            offset = attn_metadata.num_prefill_tokens
            nd = attn_metadata.num_decode_tokens
            output[offset : offset + nd] = triton_attention(
                q=query[offset : offset + nd],
                k=query[:0],  # dummy, not used in paged mode
                v=query[:0],
                b_start_loc=dm.query_start_loc,
                b_seq_len=torch.ones(nd, dtype=torch.int32, device=query.device),
                max_input_len=1,
                is_causal=True,
                softmax_scale=sm_scale,
                b_seq_len_k=dm.seq_lens,
                max_input_len_k=int(dm.seq_lens.max().item()),
                block_table=dm.block_tables,
                **paged_kw,
                **sparse_kw,
            )

        return output
