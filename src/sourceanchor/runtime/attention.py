"""Native attention capture and control implementation.

Replaces external NTIP2P dependency with diffusers-native attention hooking.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline


class AttentionStore:
    """Captures and stores cross-attention maps during UNet forward passes.

    This class accumulates attention weights across multiple denoising steps,
    allowing analysis of how text tokens influence different spatial regions.
    """

    def __init__(self):
        self.step_store: dict[str, list[torch.Tensor]] = self._get_empty_store()
        self.attention_store: dict[str, list[torch.Tensor]] = {}
        self.cur_step: int = 0

    @staticmethod
    def _get_empty_store() -> dict[str, list[torch.Tensor]]:
        """Create empty storage structure for attention maps."""
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def __call__(self, attn: torch.Tensor, is_cross: bool, place_in_unet: str) -> torch.Tensor:
        """Capture attention tensor during forward pass.

        Args:
            attn: Attention weights tensor, shape (batch*heads, spatial, seq_len)
            is_cross: Whether this is cross-attention (text-to-image)
            place_in_unet: Location string ("down", "mid", or "up")

        Returns:
            attn: Unmodified attention tensor (pass-through)
        """
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # Only store attention for reasonable resolutions (≤32x32)
        if attn.shape[1] <= 32 ** 2:
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        """Accumulate step_store into attention_store (called after each step)."""
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self._get_empty_store()
        self.cur_step += 1

    def get_average_attention(self) -> dict[str, list[torch.Tensor]]:
        """Get attention maps averaged across all accumulated steps.

        Returns:
            Dictionary mapping location keys to lists of averaged attention tensors.
        """
        if self.cur_step == 0:
            return self.step_store

        averaged = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        for key, current_items in self.step_store.items():
            if current_items:
                averaged[key] = current_items
        return averaged

    def reset(self):
        """Clear all stored attention and reset counter."""
        self.step_store = self._get_empty_store()
        self.attention_store = {}
        self.cur_step = 0


class AttentionCaptureProcessor:
    """Custom attention processor that captures cross-attention maps.

    This processor wraps the standard attention computation and optionally
    captures attention weights for later analysis.
    """

    def __init__(self, attention_store: AttentionStore | None, place_in_unet: str):
        """Initialize attention processor.

        Args:
            attention_store: Store to capture attention maps, or None to disable
            place_in_unet: Location string ("down", "mid", or "up")
        """
        self.attention_store = attention_store
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process attention computation and optionally capture weights.

        Args:
            attn: Attention module instance
            hidden_states: Input tensor
            encoder_hidden_states: Conditioning tensor (None for self-attention)
            attention_mask: Optional attention mask

        Returns:
            Output tensor after attention computation
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Determine if this is cross-attention
        is_cross = encoder_hidden_states is not None

        # Set up context for cross vs self attention
        if is_cross:
            encoder_hidden_states = encoder_hidden_states
        else:
            encoder_hidden_states = hidden_states

        # Compute Q, K, V projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention weights
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Capture attention weights if store is provided
        if self.attention_store is not None:
            self.attention_store(attention_probs, is_cross, self.place_in_unet)

        # Apply attention to values
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Project back to original dimension
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(model: StableDiffusionPipeline, controller: AttentionStore | None) -> None:
    """Register custom attention processors on UNet to capture attention maps.

    Args:
        model: StableDiffusionPipeline with unet attribute
        controller: AttentionStore to capture attention, or None to restore defaults
    """
    unet = model.unet

    if controller is None:
        # Restore default attention processors using stored defaults
        # Check if the editor stored default processors
        if hasattr(model, '_default_attn_processors'):
            unet.set_attn_processor(dict(model._default_attn_processors))
        else:
            # Fallback: manually set default processors
            try:
                unet.set_default_attn_processor()
            except ValueError:
                # If set_default fails, extract current non-custom processors
                from diffusers.models.attention_processor import AttnProcessor
                default_procs = {}
                for name in unet.attn_processors.keys():
                    default_procs[name] = AttnProcessor()
                unet.set_attn_processor(default_procs)
        return

    # Install custom processors for each attention layer
    attn_procs = {}
    for name in unet.attn_processors.keys():
        # Determine location from layer name
        if "down" in name:
            place_in_unet = "down"
        elif "mid" in name:
            place_in_unet = "mid"
        elif "up" in name:
            place_in_unet = "up"
        else:
            place_in_unet = "mid"  # fallback

        attn_procs[name] = AttentionCaptureProcessor(controller, place_in_unet)

    unet.set_attn_processor(attn_procs)


# Utility module object for compatibility
class PTPUtils:
    """Utility namespace for attention control functions."""
    register_attention_control = staticmethod(register_attention_control)


ptp_utils = PTPUtils()
