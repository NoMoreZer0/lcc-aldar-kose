from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttentionProcessor


# -------------------------------------------------------------------------
# Configuration & Context
# -------------------------------------------------------------------------

@dataclass
class SharedAttentionConfig:
    """
    Hyperparameters for cross-frame shared attention.

    consistency_strength: 0..1 — overall strength of identity sharing.
    self_keep_alpha: fraction of self features to keep when mixing K/V.
    dropout: 0..1 — stochastic dropout applied to subject-token mask.
    subject_token_top_p: top-p proportion of text tokens treated as "subject".
    subject_patch_top_p: top-p proportion of spatial positions treated as subject region.
    feature_injection_weight: blend factor for patch-level correspondence injection.
    max_patches_for_correspondence: cap on subject patches per frame for matching.
    store_attention: if True, store diagnostics (token/pos maps) in context.attn_debug.
    """
    consistency_strength: float = 0.5
    self_keep_alpha: float = 0.7
    dropout: float = 0.1
    subject_token_top_p: float = 0.15
    subject_patch_top_p: float = 0.3
    feature_injection_weight: float = 0.5
    max_patches_for_correspondence: int = 64
    store_attention: bool = False


@dataclass
class SharedAttentionContext:
    """
    Shared state passed to all attention processors for a single generation run.
    """
    cfg: SharedAttentionConfig
    step_index: int = 0
    batch_size: int = 1
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    attn_debug: Dict[str, Any] = field(default_factory=dict)
    rng: Optional[torch.Generator] = None  # RNG for dropout

    def next_step(self) -> None:
        self.step_index += 1

    def reset(self) -> None:
        self.step_index = 0
        self.attn_debug.clear()


# -------------------------------------------------------------------------
# Small Utilities
# -------------------------------------------------------------------------

def _top_k_mask_per_row(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return a boolean mask marking top-k entries per row of shape (B, L)."""
    B, L = values.shape
    if k <= 0:
        return torch.zeros((B, L), dtype=torch.bool, device=values.device)
    if k >= L:
        return torch.ones((B, L), dtype=torch.bool, device=values.device)
    _, idx = torch.topk(values, k=k, dim=-1)
    mask = torch.zeros((B, L), dtype=torch.bool, device=values.device)
    mask.scatter_(-1, idx, True)
    return mask


def _mean_of_others(x: torch.Tensor) -> torch.Tensor:
    """
    For batch tensor x (B, ...), return tensor y where y[i] = mean(x[j] for j != i).
    If B == 1, just return x.
    """
    B = x.shape[0]
    if B <= 1:
        return x
    total = x.sum(dim=0, keepdim=True)
    return (total - x) / (B - 1)


def _aggregate_token_scores(attn_probs: torch.Tensor) -> torch.Tensor:
    """Aggregate token importance per batch: (B, H, S, T) -> (B, T)."""
    return attn_probs.mean(dim=1).mean(dim=1)  # avg over heads and spatial queries


def _subject_token_mask(attn_probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Choose subject-relevant text tokens via top-p on aggregated token scores.
    (B, H, S, T) -> mask (B, T)
    """
    scores = _aggregate_token_scores(attn_probs)  # (B, T)
    B, T = scores.shape
    k = max(1, int(math.ceil(T * max(0.0, min(1.0, top_p)))))
    return _top_k_mask_per_row(scores, k)


def _subject_pos_mask(attn_probs: torch.Tensor, token_mask: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Choose subject-relevant spatial positions (HW) based on attention to subject tokens.
    attn_probs: (B, H, S, T), token_mask: (B, T) -> (B, S)
    """
    B, H, S, T = attn_probs.shape
    masked = attn_probs * token_mask.unsqueeze(1).unsqueeze(1).to(attn_probs.dtype)  # (B,H,S,T)
    pos_scores = masked.sum(dim=-1).mean(dim=1)  # (B, S)
    k = max(1, int(math.ceil(S * max(0.0, min(1.0, top_p)))))
    return _top_k_mask_per_row(pos_scores, k)


def _reshape_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    """(B, L, inner) -> (B, H, L, D)."""
    B, L, _ = x.shape
    return x.view(B, L, heads, head_dim).permute(0, 2, 1, 3)


def _square_size(n: int) -> Optional[int]:
    """Return side length s if n is a perfect square (s*s == n), else None."""
    s = int(math.sqrt(n))
    return s if s * s == n else None


# -------------------------------------------------------------------------
# Cross-Frame Attention Processor
# -------------------------------------------------------------------------

class CrossFrameAttnProcessor(torch.nn.Module):
    """
    Training-free, inference-time processor that encourages consistent subject identity
    across frames in a batch by:
      1) Selecting subject-relevant text tokens and spatial positions.
      2) Mixing cross-attention keys/values across frames for those tokens.
      3) Optionally injecting patch-level feature correspondences within subject regions.

    Applies to SDXL UNet cross-attention modules ("attn2"). Self-attention is passed through.
    """

    def __init__(self, ctx: SharedAttentionContext) -> None:
        super().__init__()
        self.ctx = ctx

    # ---- Core Call --------------------------------------------------------

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None
        if not is_cross or self.ctx.cfg.consistency_strength <= 0.0 or self.ctx.batch_size <= 1:
            return self._vanilla(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        # 1) Preprocess inputs and project Q/K/V
        residual, input_ndim, B, C, H, W, q, k, v = self._project_qkv(attn, hidden_states, encoder_hidden_states, attention_mask, temb)

        # 2) Compute attention probabilities (before mixing) for selection
        # get_attention_scores expects (B*H, L, D) format, so we need to reshape from (B, H, L, D)
        B_orig, num_heads, S, head_dim = q.shape
        _, _, T, _ = k.shape

        q_batched = q.reshape(B_orig * num_heads, S, head_dim)
        k_batched = k.reshape(B_orig * num_heads, T, head_dim)
        v_batched = v.reshape(B_orig * num_heads, T, head_dim)

        # For cross-attention, we should not pass attention_mask as it may have incompatible dimensions
        attn_probs_batched = attn.get_attention_scores(q_batched, k_batched, None)  # (B*heads, S, T)

        # Reshape back to (B, heads, S, T) for mask selection
        attn_probs = attn_probs_batched.reshape(B_orig, num_heads, S, T)

        # 3) Select subject tokens and positions
        subj_tok_mask = _subject_token_mask(attn_probs, self.ctx.cfg.subject_token_top_p)  # (B, T)
        subj_pos_mask = _subject_pos_mask(attn_probs, subj_tok_mask, self.ctx.cfg.subject_patch_top_p)  # (B, S)

        # 4) Apply dropout to token mask to prevent collapse
        subj_tok_mask = self._apply_mask_dropout(subj_tok_mask, self.ctx.cfg.dropout)

        # 5) Mix K/V across frames for subject tokens (operates on (B, H, T, D) format)
        k, v = self._mix_kv_across_frames(k, v, subj_tok_mask)

        # 6) Recompute attention with mixed keys and get per-head context
        k_batched = k.reshape(B_orig * num_heads, T, head_dim)
        v_batched = v.reshape(B_orig * num_heads, T, head_dim)
        attn_probs_batched = attn.get_attention_scores(q_batched, k_batched, None)  # (B*heads, S, T)
        context_batched = torch.matmul(attn_probs_batched, v_batched)  # (B*heads, S, D)

        # Reshape back to (B, heads, S, D) for subsequent processing
        context = context_batched.reshape(B_orig, num_heads, S, head_dim)
        attn_probs = attn_probs_batched.reshape(B_orig, num_heads, S, T)

        # 7) Optional patch-level correspondence injection within subject regions
        context = self._inject_patch_correspondence(context, attn_probs, subj_tok_mask, subj_pos_mask)

        # 8) Project back to output shape and add residual
        out = self._project_output(attn, context, residual, input_ndim, B, C, H, W)

        # 9) Optional diagnostics
        if self.ctx.cfg.store_attention:
            self._store_debug(attn, attn_probs, subj_tok_mask)

        return out

    # ---- Steps (helpers) --------------------------------------------------

    def _project_qkv(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        temb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, int, int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize, flatten (if 4D), group-norm, and project Q/K/V.
        Return: residual, input_ndim, B, C, H, W, q, k, v
        """
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)  # (B, S, C)
        else:
            B, S, C = hidden_states.shape
            H = W = int(math.sqrt(S))

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Cross context
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross is not None and kv_states is not hidden_states:
            kv_states = attn.norm_cross(kv_states)

        # Linear projections
        query = attn.to_q(hidden_states)      # (B, S, inner)
        key   = attn.to_k(kv_states)          # (B, T, inner)
        value = attn.to_v(kv_states)          # (B, T, inner)

        # Split into heads (modern diffusers >=0.26)
        # The UNet attention modules define `heads` and `dim_head` explicitly.
        if hasattr(attn, "heads") and hasattr(attn, "dim_head"):
            heads = attn.heads
            head_dim = attn.dim_head
        elif hasattr(attn, "num_heads") and hasattr(attn, "head_dim"):
            heads = attn.num_heads
            head_dim = attn.head_dim
        elif hasattr(attn, "to_q") and hasattr(attn.to_q, "weight"):
            # Fallback: infer from projection weights
            inner_dim = attn.to_q.weight.shape[0]
            heads = getattr(attn, "heads", 8)
            head_dim = inner_dim // heads
        else:
            raise AttributeError(
                f"Cannot determine attention head configuration for {attn.__class__.__name__}. "
                "Expected attributes `heads`/`dim_head` or `num_heads`/`head_dim` (diffusers >=0.26)."
            )

        # --- resolve heads & head_dim robustly across diffusers versions ---
        inner_dim = query.shape[-1]
        heads = getattr(attn, "heads", None)
        if heads is None:
            heads = getattr(attn, "num_heads", None)
        if heads is None:
            # last resort: assume 8 heads if not present (rare), adjust if your model differs
            heads = 8

        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(attn, "attention_head_dim", None)
        if head_dim is None:
            head_dim = getattr(attn, "dim_head", None)
        if head_dim is None:
            # final fallback: infer from projection width
            head_dim = inner_dim // heads
        # -------------------------------------------------------------------

        # reshape to (B, H, L, D)
        def _reshape(x: torch.Tensor) -> torch.Tensor:
            Bx, L, _ = x.shape
            # ensure inner_dim matches heads * head_dim
            expected_inner = heads * head_dim
            if x.shape[-1] != expected_inner:
                # adjust head_dim dynamically if mismatch occurs
                head_dim_adj = x.shape[-1] // heads
                return x.view(Bx, L, heads, head_dim_adj).permute(0, 2, 1, 3)
            return x.view(Bx, L, heads, head_dim).permute(0, 2, 1, 3)

        q = _reshape(query)
        k = _reshape(key)
        v = _reshape(value)

        # Do not truncate q/k/v — spatial (S) and text (T) lengths may differ in cross-attention
        return residual, input_ndim, B, C, H, W, q, k, v

    def _apply_mask_dropout(self, token_mask: torch.Tensor, p: float) -> torch.Tensor:
        """Randomly drop some subject tokens to avoid layout collapse."""
        if p <= 0.0:
            return token_mask
        keep_prob = 1.0 - p
        keep = torch.full_like(token_mask, keep_prob, dtype=torch.float32, device=token_mask.device)
        keep = torch.bernoulli(keep, generator=self.ctx.rng).to(dtype=torch.bool)
        return token_mask & keep

    def _mix_kv_across_frames(self, k: torch.Tensor, v: torch.Tensor, token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Blend K/V with the mean of other batch elements—but only for subject tokens.
        """
        others_weight = (1.0 - self.ctx.cfg.self_keep_alpha) * self.ctx.cfg.consistency_strength
        if others_weight <= 0.0 or k.shape[0] <= 1:
            return k, v

        others_k = _mean_of_others(k)  # (B, H, T, D)
        others_v = _mean_of_others(v)  # (B, H, T, D)

        mask = token_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
        mixed_k = torch.where(mask, (1 - others_weight) * k + others_weight * others_k, k)
        mixed_v = torch.where(mask, (1 - others_weight) * v + others_weight * others_v, v)
        return mixed_k, mixed_v

    def _inject_patch_correspondence(
        self,
        context: torch.Tensor,         # (B, heads, S, D)
        attn_probs: torch.Tensor,      # (B, heads, S, T)
        subj_tok_mask: torch.Tensor,   # (B, T)
        subj_pos_mask: torch.Tensor,   # (B, S)
    ) -> torch.Tensor:
        """
        Optional patch-level correspondence injection within subject regions.
        Works on head-averaged features for clarity.
        """
        weight = self.ctx.cfg.consistency_strength * self.ctx.cfg.feature_injection_weight
        if weight <= 0.0 or context.shape[0] <= 1:
            return context

        B, H, S, D = context.shape

        # Subject-related position scores: attention to subject tokens only
        masked = attn_probs.masked_fill(~subj_tok_mask.unsqueeze(1).unsqueeze(1), 0.0)  # (B,H,S,T)
        pos_scores = masked.sum(dim=-1).mean(dim=1)  # (B, S)

        # Head-averaged features per position for matching
        feat = context.mean(dim=1)                    # (B, S, D)
        feat_norm = F.normalize(feat, dim=-1)         # cosine space

        # Choose top subject positions per frame (cap for compute)
        S_top = max(1, min(self.ctx.cfg.max_patches_for_correspondence,
                           int(math.ceil(S * max(0.0, min(1.0, self.ctx.cfg.subject_patch_top_p))))))

        # Prepare blended result
        blended = feat.clone()  # (B, S, D)

        # For each frame, match its top-k subject patches to others'
        for b in range(B):
            # Select indices for this frame
            _, idx = torch.topk(pos_scores[b], k=S_top, dim=-1)  # (K,)
            src_feat = feat[b, idx]                              # (K, D)

            # Build others' bank
            others_bank = []
            for o in range(B):
                if o == b:
                    continue
                _, idx_o = torch.topk(pos_scores[o], k=S_top, dim=-1)
                others_bank.append(feat_norm[o, idx_o])         # normalized
            if not others_bank:
                continue

            bank = torch.cat(others_bank, dim=0)                # (K*(B-1), D)
            sims = torch.matmul(F.normalize(src_feat, dim=-1), bank.t())  # (K, K*(B-1))
            best = bank[sims.argmax(dim=-1)]                    # (K, D) normalized

            # Blend: keep magnitude of source, mix direction with matched
            blended[b, idx] = (1.0 - weight) * src_feat + weight * best

        # Expand back to per-head context by broadcasting blended features
        return context * 0.0 + blended.unsqueeze(1)  # (B, H, S, D)

    def _project_output(
        self,
        attn,
        context: torch.Tensor,   # (B, heads, S, D)
        residual: torch.Tensor,
        input_ndim: int,
        B: int, C: int, H: int, W: int,
    ) -> torch.Tensor:
        """
        Merge heads, project out, reshape to original spatial dims, and add residual.
        """
        # Robustly get heads and head_dim across different diffusers versions
        heads = getattr(attn, "heads", None)
        if heads is None:
            heads = getattr(attn, "num_heads", 8)

        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = getattr(attn, "attention_head_dim", None)
        if head_dim is None:
            head_dim = getattr(attn, "dim_head", None)
        if head_dim is None:
            # Infer from context tensor shape
            head_dim = context.shape[-1]

        out = context.permute(0, 2, 1, 3).reshape(B, -1, heads * head_dim)  # (B, S, inner)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1 and attn.to_out[1] is not None:
            out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(B, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out

    def _store_debug(self, attn, attn_probs: torch.Tensor, subj_tok_mask: torch.Tensor) -> None:
        """
        Store token importance and subject position maps (square if possible) for this step.
        """
        try:
            token_scores = _aggregate_token_scores(attn_probs).detach().float().cpu()  # (B, T)
            masked = attn_probs.masked_fill(~subj_tok_mask.unsqueeze(1).unsqueeze(1), 0.0)
            pos_scores = masked.sum(dim=-1).mean(dim=1).detach().float().cpu()         # (B, S)

            side = _square_size(pos_scores.shape[-1])
            if side is not None:
                pos_maps = pos_scores.view(pos_scores.shape[0], side, side)
            else:
                pos_maps = pos_scores.unsqueeze(-1)  # (B, S, 1) fallback

            self.ctx.attn_debug.setdefault("steps", {}).setdefault(self.ctx.step_index, []).append(
                {
                    "layer": getattr(attn, "_name", "attn2"),
                    "token_scores": token_scores,
                    "pos_maps": pos_maps,
                }
            )
        except Exception:
            # Debug info is best-effort; never break generation.
            pass

    # ---- Vanilla attention (no sharing) -----------------------------------

    @staticmethod
    def _vanilla(attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        """Close reproduction of diffusers' default attention processor."""
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)  # (B, S, C)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross is not None and kv is not hidden_states:
            kv = attn.norm_cross(kv)
        key = attn.to_k(kv)
        value = attn.to_v(kv)

        heads, head_dim = attn.heads, attn.head_dim
        q = _reshape_heads(query, heads, head_dim)
        k = _reshape_heads(key,   heads, head_dim)
        v = _reshape_heads(value, heads, head_dim)

        is_cross = encoder_hidden_states is not None
        mask = None if is_cross else attention_mask
        attn_probs = attn.get_attention_scores(q, k, mask)
        context = torch.matmul(attn_probs, v)

        out = context.permute(0, 2, 1, 3).reshape(q.shape[0], -1, heads * head_dim)
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1 and attn.to_out[1] is not None:
            out = attn.to_out[1](out)

        if input_ndim == 4:
            out = out.transpose(1, 2).reshape(B, C, H, W)
        if attn.residual_connection:
            out = out + residual
        return out


# -------------------------------------------------------------------------
# Apply / Restore helpers
# -------------------------------------------------------------------------

def apply_shared_attention_to_unet(
    unet,
    ctx: SharedAttentionContext,
    only_cross_attention: bool = True,
) -> Dict[str, AttentionProcessor]:
    """
    Replace UNet attention processors with CrossFrameAttnProcessor where desired.
    Returns original processors for later restoration.
    """
    original = unet.attn_processors
    new_mapping: Dict[str, AttentionProcessor] = {}
    for name, proc in original.items():
        is_cross = "attn2" in name  # diffusers naming convention
        new_mapping[name] = proc if (only_cross_attention and not is_cross) else CrossFrameAttnProcessor(ctx)
    unet.set_attn_processor(new_mapping)
    return original


def restore_unet_attention(unet, prev_processors: Dict[str, AttentionProcessor]) -> None:
    """Restore previously saved UNet attention processors."""
    unet.set_attn_processor(prev_processors)


# -------------------------------------------------------------------------
# Debug image export
# -------------------------------------------------------------------------

def save_attention_debug_pngs(attn_debug: Dict[str, Any], save_dir: "PathLike") -> List[str]:
    """
    Save per-step subject position maps to PNGs for quick visualization.
    Returns a list of written file paths.
    """
    from pathlib import Path
    from PIL import Image
    import numpy as np

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    steps = attn_debug.get("steps", {})
    for step, items in steps.items():
        if not items:
            continue
        entry = items[-1]  # take last recorded layer for that step
        pos_maps = entry.get("pos_maps")
        if pos_maps is None:
            continue

        arr = pos_maps.numpy()  # (B, H, W) or (B, S, 1)
        B = arr.shape[0]
        for b in range(B):
            img = arr[b]
            # Normalize to 0..255
            vmin, vmax = img.min(), img.max()
            norm = (img - vmin) / (max(vmax - vmin, 1e-8))
            norm = (norm * 255).astype(np.uint8)
            if norm.ndim == 3 and norm.shape[-1] == 1:
                norm = norm[..., 0]
            pil = Image.fromarray(norm)
            path = out_dir / f"attn_step{step:03d}_frame{b:02d}.png"
            pil.save(path)
            written.append(str(path))
    return written
