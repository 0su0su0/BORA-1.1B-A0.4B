# BORA-1.1B-A0.4B: 1.1B parameter (366M active) EMA-Attention Hybrid with Sparse MoE
#   - Local: EMA time mixing (RWKV-inspired, fast parallel scan)
#   - Global: NSA (Native Sparse Attention, hardware-optimized)
#   - MoE: 16 sparse + 2 shared experts, top-2 routing
# License: Apache 2.0 compatible
from dataclasses import dataclass
from typing import Any, List, Optional

import math
import mlx.core as mx
import mlx.nn as nn

# Metal kernel for EMA (optional - falls back to vanilla if import fails)
try:
    from ema_metal import ema_hybrid, ema_vanilla
    EMA_METAL_AVAILABLE = True
except ImportError:
    EMA_METAL_AVAILABLE = False

from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bora_moe"
    vocab_size: int = 32128
    hidden_size: int = 1024
    num_hidden_layers: int = 23

    # Attention (for SparseAttn layers)
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    norm_eps: float = 1e-5

    # EMA parameters
    d_conv: int = 4  # Convolution kernel size for local patterns
    expand: int = 2  # Expansion factor for inner dimension
    ema_decay_init: float = 0.9  # Initial decay rate for EMA

    # Sparse MLP (MoE)
    num_sparse_experts: int = 16
    num_shared_experts: int = 2
    num_experts_per_tok: int = 2
    ffn_dim: int = 768
    gate_activation: str = "sigmoid"
    adaptive_routing: bool = True
    normalize_weights: bool = True

    # NSA parameters (for SparseAttn)
    compress_block: int = 8
    select_block: int = 32
    num_selected: int = 16
    sliding_window: int = 512

    # Layer pattern (required in config)
    layer_types: Optional[List[str]] = None
    attn_layer_idxs: Optional[List[int]] = None
    dense_layer_idxs: Optional[List[int]] = None

    # Training optimization
    grad_checkpoint: bool = False

    def __post_init__(self):
        # Validate required fields
        if self.layer_types is None:
            raise ValueError("layer_types is required for BORA-1.1B-A0.4B")
        if self.dense_layer_idxs is None:
            raise ValueError("dense_layer_idxs is required for BORA-1.1B-A0.4B")

        # Derive attn_layer_idxs from layer_types
        if self.attn_layer_idxs is None:
            self.attn_layer_idxs = [
                i for i, t in enumerate(self.layer_types) if t == "attn"
            ]


class EMACache:
    """Cache for EMA inference (conv state + EMA hidden state)."""

    def __init__(self, batch_size: int, d_inner: int, d_conv: int):
        self.conv_state = mx.zeros((batch_size, d_conv - 1, d_inner))
        self.ema_state = mx.zeros((batch_size, d_inner))

    @property
    def state(self):
        return (self.conv_state, self.ema_state)


class EMAMixer(nn.Module):
    """MEGA-style Damped EMA time mixing.

    Based on MEGA (arXiv:2209.10655) Multi-dimensional Damped EMA.

    Recurrence: h[t] = φ * h[t-1] + α * x[t]
    where φ = 1 - α·δ (α = decay, δ = damping)

    - δ→1: standard EMA (sum of coefficients = 1)
    - δ<1: state preservation enhanced (sum > 1, residual effect)
    - φ ∈ (0,1) always guaranteed (both α,δ are sigmoid)
    """

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = args.hidden_size
        self.d_conv = args.d_conv
        self.expand = args.expand
        self.d_inner = self.d_model * self.expand

        # Input projection: d_model -> 2 * d_inner (for x and gate branches)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)

        # Short convolution for local patterns
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            bias=True,
        )

        # Learnable per-channel decay α (in logit space for sigmoid)
        decay_init = args.ema_decay_init  # 0.9
        decay_logit = math.log(decay_init / (1 - decay_init))
        self.decay_logit = mx.full((self.d_inner,), decay_logit)

        # Learnable per-channel damping δ (MEGA paper)
        # Init δ≈0.999 → starts as standard EMA, learns to adjust
        damping_init = 0.999
        damping_logit = math.log(damping_init / (1 - damping_init))
        self.damping_logit = mx.full((self.d_inner,), damping_logit)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _ema_parallel(self, x: mx.array, decay: mx.array, damping: mx.array) -> mx.array:
        """Parallel Damped EMA via cumsum trick (fallback).

        Damped EMA: h[t] = φ * h[t-1] + α * x[t]
        where φ = 1 - α·δ

        Solution: h[t] = α * Σ_{i=0}^{t} φ^{t-i} * x[i]
                       = α * φ^t * cumsum(x / φ^i)[t]
        """
        B, L, ED = x.shape
        phi = 1 - decay * damping  # state coefficient

        log_phi = mx.log(phi + 1e-8)
        t = mx.arange(L)[:, None]
        log_phi_pow_t = t * log_phi[None, :]

        phi_pow_t = mx.exp(log_phi_pow_t)
        phi_pow_neg_t = mx.exp(-log_phi_pow_t)

        y = x * phi_pow_neg_t[None, :, :]
        y_cumsum = mx.cumsum(y, axis=1)
        h = decay[None, None, :] * phi_pow_t[None, :, :] * y_cumsum

        return h

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None,
        pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Args:
            x: (B, L, D)
            mask: Not used (handles causality internally)
            cache: EMACache for inference
            pad_mask: (B, L) bool — True for real tokens, False for padding

        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection: (B, L, D) -> (B, L, 2*ED)
        xz = self.in_proj(x)
        x_branch, z = mx.split(xz, 2, axis=-1)  # Each: (B, L, ED)

        # Convolution (causal padding) for local patterns
        if cache is not None:
            x_conv = mx.concatenate([cache.conv_state, x_branch], axis=1)
            cache.conv_state = x_conv[:, -self.d_conv + 1:, :]
        else:
            x_conv = mx.pad(x_branch, [(0, 0), (self.d_conv - 1, 0), (0, 0)])

        x_conv = self.conv1d(x_conv)  # (B, L, ED)
        x_conv = nn.silu(x_conv)

        # Padding mask: zero out padding tokens before EMA (MEGA paper)
        if pad_mask is not None:
            x_conv = x_conv * pad_mask[:, :, None]  # (B,L,1) broadcast

        # Get decay α and damping δ
        decay = mx.sigmoid(self.decay_logit)    # α ∈ (0,1)
        damping = mx.sigmoid(self.damping_logit)  # δ ∈ (0,1)

        # Damped EMA: h[t] = φ*h[t-1] + α*x[t], φ = 1-α·δ
        if cache is not None:
            phi = 1 - decay * damping
            h = phi * cache.ema_state[:, None, :] + decay * x_conv
            cache.ema_state = h[:, -1, :]
        else:
            if EMA_METAL_AVAILABLE:
                h = ema_vanilla(x_conv, decay, damping)
            else:
                h = self._ema_parallel(x_conv, decay, damping)

        # Gate with z branch
        y = h * nn.silu(z)

        # Output projection
        return self.out_proj(y)


class SparseAttn(nn.Module):
    """NSA-Lite: 3-branch sparse attention.

    Based on Native Sparse Attention (DeepSeek, 2025).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # QK normalization
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.norm_eps)

        # RoPE
        self.rope = nn.RoPE(self.head_dim, base=args.rope_theta, traditional=False)

        # NSA parameters
        self.compress_block = args.compress_block
        self.num_selected = args.num_selected
        self.sliding_window = args.sliding_window

        # Compression projection (per-head — NSA paper)
        self.kv_compress = nn.Linear(
            self.head_dim * self.compress_block,
            self.head_dim,
            bias=False
        )

        # Branch gate (per-head, sigmoid — NSA paper)
        self.branch_gate = nn.Linear(dim, self.n_heads * 3, bias=False)

        # Block selection group size (NSA paper)
        self.select_block = args.select_block

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None,
        pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, T, D = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and normalize
        q = self.q_norm(q.reshape(B, T, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(k.reshape(B, T, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Apply RoPE
        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        T_kv = k.shape[2]

        # Standard attention for short sequences or with cache
        if T_kv <= self.sliding_window or cache is not None:
            if cache is not None:
                # Inference: T_q may differ from T_kv
                causal_mask = mx.triu(mx.full((T, T_kv), -mx.inf), k=T_kv - T + 1)
                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=causal_mask
                )
            else:
                # Training short seq: T_q == T_kv
                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask="causal"
                )
        else:
            # NSA 3-branch for long sequences
            output = self._nsa_forward(q, k, v, x, T, T_kv)

        output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(output)

    def _nsa_forward(self, q, k, v, x, T, T_kv):
        """NSA 3-branch forward (per-query-group block selection — NSA paper).

        Optimized: mx.repeat 제거, SDPA + broadcast matmul 사용.
        """
        B = x.shape[0]
        kv_repeat = self.n_heads // self.n_kv_heads
        block = self.compress_block    # 16
        sel_block = self.select_block  # 32

        # --- Branch 1: Compressed attention (SDPA, no repeat) ---
        k_comp, v_comp = self._compress_kv(k, v)
        n_blocks = k_comp.shape[2]

        # Block-level causal mask (2D, head-independent)
        block_ends = (mx.arange(n_blocks) + 1) * block - 1
        comp_mask = mx.where(
            block_ends[None, :] <= mx.arange(T)[:, None], 0.0, -1e9
        )

        # Importance scores: 5D broadcast matmul (no repeat)
        q_for_imp = q.reshape(B, self.n_kv_heads, kv_repeat, T, self.head_dim)
        scores_imp = (
            q_for_imp @ k_comp[:, :, None, :, :].transpose(0, 1, 2, 4, 3)
        ) * self.scale
        scores_imp = scores_imp + comp_mask  # broadcast (T, n_blocks)
        importance = scores_imp.sum(axis=2)  # (B, n_kv, T, n_blocks)

        # Compressed output: SDPA (GQA native, no repeat)
        out_compressed = mx.fast.scaled_dot_product_attention(
            q, k_comp, v_comp, scale=self.scale, mask=comp_mask
        )

        # --- Branch 2: Per-query-group selected attention (broadcast matmul, no repeat) ---
        # Step A-B: group importance by select_block
        n_groups = T // sel_block
        importance_grouped = importance[:, :, :n_groups * sel_block, :].reshape(
            B, self.n_kv_heads, n_groups, sel_block, n_blocks
        ).mean(axis=3)  # (B, n_kv, n_groups, n_blocks)

        # Step C: per-group top-k
        importance_sum = importance_grouped.sum(axis=1)  # (B, n_groups, n_blocks)
        k_sel = min(self.num_selected, n_blocks)
        top_indices = mx.argpartition(
            importance_sum, kth=-k_sel, axis=-1
        )[..., -k_sel:]  # (B, n_groups, num_selected)

        # Step D: per-group KV gather (no repeat)
        k_sel_blocks, v_sel_blocks = self._gather_blocks_per_group(
            k, v, top_indices, T_kv
        )  # (B, n_kv, n_groups, sel_len, D)

        # Step E: grouped attention via broadcast matmul (no repeat)
        # Q: (B, n_heads, T, D) → (B, n_kv, repeat, n_groups, sel_block, D)
        q_grouped = q[:, :, :n_groups * sel_block, :].reshape(
            B, self.n_kv_heads, kv_repeat, n_groups, sel_block, self.head_dim
        )
        # K/V: (B, n_kv, n_groups, sel_len, D) → broadcast via [:, :, None]
        scores_sel = (
            q_grouped @ k_sel_blocks[:, :, None, :, :, :].transpose(0, 1, 2, 3, 5, 4)
        ) * self.scale

        # Causal mask: track original positions
        block_starts = top_indices * block  # (B, n_groups, num_selected)
        offsets = mx.arange(block)
        orig_pos = (block_starts[:, :, :, None] + offsets[None, None, None, :]).reshape(
            B, n_groups, -1
        )  # (B, n_groups, sel_len)
        q_pos = mx.arange(n_groups)[:, None] * sel_block + mx.arange(sel_block)[None, :]
        sel_mask = mx.where(
            orig_pos[:, None, :, None, :] <= q_pos[None, None, :, :, None],
            0.0, -1e9
        )  # (B, 1, n_groups, sel_block, sel_len)
        scores_sel = scores_sel + sel_mask[:, :, None, :, :, :]  # broadcast repeat dim
        out_selected_grouped = mx.softmax(scores_sel, axis=-1) @ v_sel_blocks[:, :, None, :, :, :]
        out_selected = out_selected_grouped.reshape(
            B, self.n_heads, n_groups * sel_block, self.head_dim
        )

        # Handle remainder tokens (T % sel_block != 0)
        if n_groups * sel_block < T:
            remainder = T - n_groups * sel_block
            out_selected = mx.concatenate([
                out_selected,
                mx.zeros((B, self.n_heads, remainder, self.head_dim))
            ], axis=2)

        # --- Branch 3: Sliding window (SDPA, no repeat) ---
        window_mask = self._sliding_mask(T, T_kv)
        out_sliding = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=window_mask
        )

        # --- Per-head sigmoid gate (NSA paper) ---
        gate_logits = self.branch_gate(x)  # (B, T, n_heads*3)
        gates = mx.sigmoid(gate_logits.reshape(B, T, self.n_heads, 3))
        gates = gates.transpose(0, 2, 1, 3)  # (B, H, T, 3)

        return (
            gates[..., 0:1] * out_compressed +
            gates[..., 1:2] * out_selected +
            gates[..., 2:3] * out_sliding
        )

    def _compress_kv(self, k, v):
        """Compress KV into block representations (per-head — NSA paper)."""
        B, n_kv, T, D = k.shape
        block = self.compress_block

        pad_len = (block - T % block) % block
        if pad_len > 0:
            k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])

        n_blocks = k.shape[2] // block
        k_blocks = k.reshape(B, n_kv, n_blocks, block, D)

        # Per-head compression for K: each (batch, head, block) independently
        k_flat = k_blocks.reshape(B * n_kv * n_blocks, block * D)
        k_comp = self.kv_compress(k_flat).reshape(B, n_kv, n_blocks, D)

        # Mean pooling for V
        v_blocks = v.reshape(B, n_kv, n_blocks, block, D)
        v_comp = v_blocks.mean(axis=3)

        return k_comp, v_comp

    def _gather_blocks_per_group(self, k, v, indices, T):
        """Per-group block gather (NSA paper). Flat indexing for memory efficiency.

        Args:
            k, v: (B, n_kv, T, D)
            indices: (B, n_groups, num_selected) block indices per group
            T: original sequence length (for padding calc)

        Returns:
            k_sel, v_sel: (B, n_kv, n_groups, num_selected*block, D)
        """
        B, n_kv, _, D = k.shape
        block = self.compress_block
        n_groups, n_sel = indices.shape[1], indices.shape[2]

        pad_len = (block - T % block) % block
        if pad_len > 0:
            k = mx.pad(k, [(0, 0), (0, 0), (0, pad_len), (0, 0)])
            v = mx.pad(v, [(0, 0), (0, 0), (0, pad_len), (0, 0)])

        n_blocks_total = k.shape[2] // block
        # Merge block and D dims for 3D gather
        k_blocks = k.reshape(B, n_kv, n_blocks_total, block * D)
        v_blocks = v.reshape(B, n_kv, n_blocks_total, block * D)

        # Flatten group indices for single gather
        idx_flat = indices.reshape(B, n_groups * n_sel)  # (B, total_gathered)
        idx_flat = idx_flat[:, None, :, None]  # (B, 1, total, 1)
        idx_flat = mx.broadcast_to(idx_flat, (B, n_kv, n_groups * n_sel, block * D))

        k_sel = mx.take_along_axis(k_blocks, idx_flat, axis=2)
        v_sel = mx.take_along_axis(v_blocks, idx_flat, axis=2)

        # Reshape to (B, n_kv, n_groups, n_sel*block, D)
        k_sel = k_sel.reshape(B, n_kv, n_groups, n_sel * block, D)
        v_sel = v_sel.reshape(B, n_kv, n_groups, n_sel * block, D)

        return k_sel, v_sel

    def _sliding_mask(self, T_q, T_kv):
        """Create sliding window causal mask."""
        mask = mx.triu(mx.full((T_q, T_kv), -mx.inf), k=T_kv - T_q + 1)
        window_mask = mx.tril(mx.full((T_q, T_kv), -mx.inf), k=T_kv - T_q - self.sliding_window)
        return mask + window_mask


class DenseMLP(nn.Module):
    """Standard SwiGLU MLP."""

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMLP(nn.Module):
    """Sparse MoE with adaptive routing."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.num_sparse = args.num_sparse_experts
        self.num_shared = args.num_shared_experts
        self.top_k = args.num_experts_per_tok
        self.normalize_weights = args.normalize_weights
        self.gate_activation = args.gate_activation
        self.adaptive_routing = args.adaptive_routing

        self.router = nn.Linear(dim, self.num_sparse, bias=False)

        if self.adaptive_routing:
            self.routing_offset = mx.zeros((self.num_sparse,))

        self.sparse_experts = SwitchGLU(dim, args.ffn_dim, self.num_sparse)
        # Single large shared expert (replaces 2 separate experts)
        self.shared_expert = DenseMLP(dim, args.ffn_dim * self.num_shared)

        self.expert_counts = mx.zeros((self.num_sparse,))
        self.expert_total = mx.array(0.0)

    def reset_expert_counts(self):
        """Reset accumulated expert counts (call at each optimizer step)."""
        self.expert_counts = mx.zeros((self.num_sparse,))
        self.expert_total = mx.array(0.0)

    def __call__(self, x: mx.array, pad_mask: Optional[mx.array] = None) -> mx.array:
        affinity = self.router(x)
        scores = mx.sigmoid(affinity)

        # Adaptive routing: argpartition에 직접 전달 (불필요한 배열 생성 제거)
        expert_indices = mx.argpartition(
            scores + self.routing_offset, kth=-self.top_k, axis=-1
        )[..., -self.top_k:]

        expert_weights = mx.take_along_axis(scores, expert_indices, axis=-1)
        expert_weights = expert_weights / (
            mx.sum(expert_weights, axis=-1, keepdims=True) + 1e-6
        )
        expert_weights = expert_weights.astype(x.dtype)

        sparse_out = self.sparse_experts(x, expert_indices)
        sparse_out = (sparse_out * expert_weights[..., None]).sum(axis=-2)

        # Expert counting (학습시에만 실행)
        if self.training:
            flat = mx.stop_gradient(expert_indices).reshape(-1)
            if pad_mask is not None:
                token_mask = mx.repeat(mx.stop_gradient(pad_mask).reshape(-1), self.top_k).astype(mx.float32)
                counts = mx.sum(
                    (flat[:, None] == mx.arange(self.num_sparse)[None, :]) * token_mask[:, None],
                    axis=0,
                ).astype(mx.float32)
                self.expert_counts = self.expert_counts + counts
                self.expert_total = self.expert_total + mx.sum(token_mask)
            else:
                counts = mx.sum(flat[:, None] == mx.arange(self.num_sparse)[None, :], axis=0).astype(mx.float32)
                self.expert_counts = self.expert_counts + counts
                self.expert_total = self.expert_total + flat.size

        shared_out = self.shared_expert(x)

        return sparse_out + shared_out


class BoraBlock(nn.Module):
    """Single BORA block: EMAMixer or SparseAttn + MoE FFN."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_attn = layer_idx in args.attn_layer_idxs

        if self.is_attn:
            self.mixer = SparseAttn(args)
        else:
            self.mixer = EMAMixer(args, layer_idx)

        if layer_idx in args.dense_layer_idxs:
            self.ffn = DenseMLP(args.hidden_size, args.ffn_dim)
        else:
            self.ffn = SparseMLP(args)

        self.mixer_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None,
        pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        h = x + self.mixer(self.mixer_norm(x), mask=mask, cache=cache, pad_mask=pad_mask)
        if pad_mask is not None and isinstance(self.ffn, SparseMLP):
            return h + self.ffn(self.ffn_norm(h), pad_mask=pad_mask)
        return h + self.ffn(self.ffn_norm(h))


class BoraModel(nn.Module):
    """BORA backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.grad_checkpoint = args.grad_checkpoint

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [BoraBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.norm_eps)

        self.first_attn_idx = args.attn_layer_idxs[0] if args.attn_layer_idxs else 0

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        # pad_mask: (B, T) - True for real tokens, False for pad (training only)
        pad_mask = None
        if input_embeddings is None and cache is None:
            pad_mask = (inputs != 0)  # pad_token_id = 0

        if cache is None:
            cache = [None] * len(self.layers)

        # Create attention mask for SparseAttn layers
        from mlx_lm.models.base import create_attention_mask
        attn_mask = create_attention_mask(h, cache[self.first_attn_idx])

        for layer, c in zip(self.layers, cache):
            # Mamba doesn't need mask, SparseAttn uses attn_mask
            mask = attn_mask if layer.is_attn else None

            if self.grad_checkpoint and c is None:
                h = mx.checkpoint(layer)(h, mask, cache=None, pad_mask=pad_mask)
            else:
                h = layer(h, mask, cache=c, pad_mask=pad_mask)

        return self.norm(h)


class Model(nn.Module):
    """BORA-1.1B-A0.4B: EMA-Attention Hybrid with Sparse MoE."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = BoraModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        return self.model.embed_tokens.as_linear(out)

    def sanitize(self, weights):
        sanitized = {}
        for name, param in weights.items():
            if "conv1d.weight" in name or "conv.weight" in name:
                if param.ndim == 3 and param.shape[-1] > param.shape[1]:
                    param = param.transpose(0, 2, 1)
            sanitized[name] = param
        return sanitized

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, batch_size: int = 1):
        caches = []
        for layer in self.layers:
            if layer.is_attn:
                caches.append(KVCache())
            else:
                caches.append(EMACache(
                    batch_size=batch_size,
                    d_inner=layer.mixer.d_inner,
                    d_conv=layer.mixer.d_conv,
                ))
        return caches

    @property
    def cast_predicate(self):
        def predicate(k):
            return ("routing_offset" not in k and "decay_logit" not in k
                    and "expert_counts" not in k and "expert_total" not in k)
        return predicate

    def get_expert_stats(self) -> dict:
        """Expert utilization stats from accumulated counts across all microbatches."""
        stats = {}
        for i, layer in enumerate(self.layers):
            ffn = layer.ffn
            if not hasattr(ffn, "expert_counts") or ffn.expert_total.item() == 0:
                continue
            counts = ffn.expert_counts
            total = ffn.expert_total.item()
            pct = counts / total * 100
            stats[f"layer_{i}"] = {
                'counts': counts,
                'min': mx.min(pct),
                'max': mx.max(pct),
                'std': mx.sqrt(mx.mean((pct - mx.mean(pct)) ** 2)),
            }
        return stats

    def update_expert_biases(self, gamma=0.001):
        """Update routing offsets using accumulated counts (DeepSeek-V3 style).

        Uses full batch statistics (all microbatches), not just the last one.

        Args:
            gamma: float (uniform) or dict[int, float] (per-layer, keyed by layer index)
        """
        per_layer = isinstance(gamma, dict)
        offsets_to_eval = []
        for i, layer in enumerate(self.layers):
            ffn = layer.ffn
            if not hasattr(ffn, "adaptive_routing") or not ffn.adaptive_routing:
                continue
            if not hasattr(ffn, "expert_counts") or ffn.expert_total.item() == 0:
                continue

            g = gamma.get(i, 0.0) if per_layer else gamma

            counts = ffn.expert_counts
            n = ffn.num_sparse
            target = ffn.expert_total.item() / n

            # DeepSeek-V3: fixed +/- gamma
            sign = mx.where(counts < target, 1.0, -1.0)
            ffn.routing_offset = ffn.routing_offset + g * sign
            offsets_to_eval.append(ffn.routing_offset)

            # Reset for next step
            ffn.reset_expert_counts()

        # Batch eval: one sync instead of per-layer
        if offsets_to_eval:
            mx.eval(*offsets_to_eval)
