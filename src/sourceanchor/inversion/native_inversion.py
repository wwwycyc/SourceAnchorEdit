"""Native DDIM inversion implementation using diffusers APIs.

Replaces external NTIP2P dependency with native PyTorch and diffusers code.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline


class NativeInversion:
    """DDIM inversion for converting real images to noise latents.

    This implementation uses diffusers' native scheduler and pipeline components
    to invert images through the diffusion process, enabling real image editing.
    """

    def __init__(self, pipe: StableDiffusionPipeline):
        """Initialize inversion backend.

        Args:
            pipe: StableDiffusionPipeline with configured components
        """
        self.pipe = pipe
        self.prompt = ""
        self.context = None  # Will store encoded prompt embeddings

        # Get device and dtype from pipeline
        self.device = getattr(pipe, "_execution_device", pipe.device)
        self.dtype = pipe.unet.dtype

    def init_prompt(self, prompt: str) -> None:
        """Encode and store prompt for inversion.

        Args:
            prompt: Text prompt (can be empty string for unconditional)
        """
        self.prompt = prompt if prompt else ""

        # Encode prompt to get embeddings
        text_inputs = self.pipe.tokenizer(
            [self.prompt],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            prompt_embeds = self.pipe.text_encoder(text_input_ids)[0]

        # For classifier-free guidance, need both conditional and unconditional
        # Create unconditional embeddings (empty prompt)
        uncond_input = self.pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_input.input_ids.to(self.device)

        with torch.no_grad():
            uncond_embeds = self.pipe.text_encoder(uncond_input_ids)[0]

        # Stack for classifier-free guidance: [uncond, cond]
        self.context = torch.cat([uncond_embeds, prompt_embeds]).to(dtype=self.dtype)

    @torch.no_grad()
    def image2latent(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """Encode image to latent space using VAE.

        Args:
            image: Input image as numpy array (H, W, 3) uint8 or PIL Image

        Returns:
            Latent tensor of shape (1, 4, H//8, W//8)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if isinstance(image, np.ndarray):
            # Convert from uint8 [0, 255] to float [-1, 1]
            image = torch.from_numpy(image).float() / 127.5 - 1.0
            # HWC -> CHW
            image = image.permute(2, 0, 1).unsqueeze(0)

        image = image.to(device=self.device, dtype=self.pipe.vae.dtype)

        # Encode with VAE
        latent_dist = self.pipe.vae.encode(image).latent_dist
        latents = latent_dist.mean  # Use mean instead of sampling
        latents = latents * self.pipe.vae.config.scaling_factor  # Scale by 0.18215

        return latents

    @torch.no_grad()
    def latent2image(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latent to image using VAE.

        Args:
            latents: Latent tensor of shape (1, 4, H//8, W//8)

        Returns:
            Image as numpy array (H, W, 3) uint8 in [0, 255]
        """
        # Unscale latents
        latents = latents / self.pipe.vae.config.scaling_factor

        # Decode with VAE
        image = self.pipe.vae.decode(latents).sample

        # Convert from [-1, 1] to [0, 1]
        image = (image / 2 + 0.5).clamp(0, 1)

        # Convert to numpy uint8
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)

        return image

    def _get_noise_pred(self, latents: torch.Tensor, t: torch.Tensor, is_forward: bool = True) -> torch.Tensor:
        """Predict noise using UNet with classifier-free guidance.

        Args:
            latents: Current latent tensor
            t: Timestep
            is_forward: Whether this is forward (inversion) or backward (generation)

        Returns:
            Predicted noise tensor
        """
        # Duplicate latents for classifier-free guidance
        latent_model_input = torch.cat([latents] * 2)

        # Predict noise
        noise_pred = self.pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=self.context,
        ).sample

        # Perform classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        # Use guidance scale (from module globals, set by configure function)
        guidance_scale = getattr(self.pipe, "_guidance_scale", 1.0) if is_forward else getattr(self.pipe, "_guidance_scale", 7.5)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    def _ddim_step_forward(self, latents: torch.Tensor, noise_pred: torch.Tensor, t: int, t_next: int) -> torch.Tensor:
        """Perform one DDIM forward step (image -> noise).

        Args:
            latents: Current latent z_t
            noise_pred: Predicted noise
            t: Current timestep
            t_next: Next timestep (should be > t for forward)

        Returns:
            Next latent z_{t+1}
        """
        # Get alpha values from scheduler
        alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = self.pipe.scheduler.alphas_cumprod[t_next] if t_next < len(self.pipe.scheduler.alphas_cumprod) else self.pipe.scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # Predict x_0 (denoised latent)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5

        # Compute direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * noise_pred

        # Compute x_{t+1}
        next_latents = alpha_prod_t_next ** 0.5 * pred_original_sample + pred_sample_direction

        return next_latents

    @torch.no_grad()
    def ddim_inversion(self, image: np.ndarray | Image.Image) -> tuple[np.ndarray, list[torch.Tensor]]:
        """Perform DDIM inversion on input image.

        Args:
            image: Input image as numpy array (H, W, 3) uint8 or PIL Image

        Returns:
            Tuple of (reconstruction_image, ddim_latents):
            - reconstruction_image: np.ndarray (H, W, 3) uint8, reconstructed from initial latent
            - ddim_latents: List of latent tensors, last element is the fully noised latent
        """
        # Encode image to latent
        latents = self.image2latent(image)

        # Reconstruct image from initial latent (before inversion) to verify encoding quality
        reconstruction_image = self.latent2image(latents)

        # Get inversion timesteps (forward in time: 0 -> T)
        num_inference_steps = getattr(self.pipe, "_num_ddim_steps", 50)
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        # Reverse timesteps for inversion (we go from clean to noisy)
        timesteps = reversed(timesteps)
        timesteps = list(timesteps)

        # Store all intermediate latents
        ddim_latents = [latents.clone()]

        # Perform inversion steps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Ensure timesteps are tensors on correct device
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device, dtype=torch.long)
            if not isinstance(t_next, torch.Tensor):
                t_next = torch.tensor(t_next, device=self.device, dtype=torch.long)

            # Predict noise using conditional embeddings only (no CFG during inversion)
            # Use only the conditional part of context
            _, cond_embeddings = self.context.chunk(2)
            noise_pred = self.pipe.unet(
                latents,
                t,
                encoder_hidden_states=cond_embeddings,
            ).sample

            # DDIM forward step
            latents = self._ddim_step_forward(latents, noise_pred, t.item() if hasattr(t, 'item') else int(t), t_next.item() if hasattr(t_next, 'item') else int(t_next))

            ddim_latents.append(latents.clone())

        return reconstruction_image, ddim_latents
