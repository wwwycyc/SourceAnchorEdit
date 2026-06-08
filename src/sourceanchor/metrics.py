"""Metrics calculation for image editing evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from torchvision.transforms import Resize
from transformers import CLIPConfig, CLIPModel, CLIPProcessor


class DinoVitExtractor:
    BLOCK_KEY = "block"
    QKV_KEY = "qkv"

    def __init__(self, model_name: str, device: str) -> None:
        try:
            self.model = torch.hub.load("facebookresearch/dino:main", model_name, trust_repo=True).to(device)
        except TypeError:
            self.model = torch.hub.load("facebookresearch/dino:main", model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device
        self.hook_handlers: list[Any] = []
        self.outputs_dict: dict[str, list[torch.Tensor]] = {}
        self._init_hooks_data()

    def _init_hooks_data(self) -> None:
        self.outputs_dict = {
            self.BLOCK_KEY: [],
            self.QKV_KEY: [],
        }

    def _register_hooks(self) -> None:
        for block in self.model.blocks:
            self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))

    def _clear_hooks(self) -> None:
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _hook(_model, _inp, output):
            self.outputs_dict[self.BLOCK_KEY].append(output)

        return _hook

    def _get_qkv_hook(self):
        def _hook(_model, _inp, output):
            self.outputs_dict[self.QKV_KEY].append(output)

        return _hook

    def get_qkv_feature_from_input(self, input_img: torch.Tensor) -> list[torch.Tensor]:
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[self.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_num(self, input_img_shape: tuple[int, ...]) -> int:
        _, _, h, w = input_img_shape
        patch_size = 8 if "8" in self.model_name else 16
        return 1 + (h // patch_size) * (w // patch_size)

    def get_head_num(self) -> int:
        return 6 if "s" in self.model_name else 12

    def get_embedding_dim(self) -> int:
        return 384 if "s" in self.model_name else 768

    def get_keys_from_qkv(self, qkv: torch.Tensor, input_img_shape: tuple[int, ...]) -> torch.Tensor:
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        return qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]

    def get_keys_from_input(self, input_img: torch.Tensor, layer_num: int) -> torch.Tensor:
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        return self.get_keys_from_qkv(qkv_features, tuple(input_img.shape))

    def get_keys_self_sim_from_input(self, input_img: torch.Tensor, layer_num: int) -> torch.Tensor:
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        return self.attn_cosine_sim(concatenated_keys[None, None, ...])

    @staticmethod
    def attn_cosine_sim(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x = x[0]
        norm = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        return (x @ x.permute(0, 2, 1)) / factor


class DinoStructureDistanceCalculator(torch.nn.Module):
    def __init__(self, model_name: str, patch_size: int, device: str) -> None:
        super().__init__()
        self.device = device
        self.extractor = DinoVitExtractor(model_name=model_name, device=device)
        imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.global_transform = transforms.Compose(
            [
                Resize(patch_size, max_size=480),
                imagenet_norm,
            ]
        )

    def calculate_global_ssim_loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)
        for a, b in zip(inputs, outputs):
            a = self.global_transform(a)
            b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_self_sim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss = loss + F.mse_loss(keys_self_sim, target_keys_self_sim)
        return loss


class MetricsCalculator:
    """Calculate evaluation metrics for edited images."""

    def __init__(
        self,
        device: str = "cuda",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        *,
        lpips_net: str = "squeeze",
        clip_local_files_only: bool = True,
        dino_model_name: str = "dino_vits8",
        dino_global_patch_size: int = 224,
    ):
        self.device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        self.clip_model_id = clip_model_id
        self.lpips_net = lpips_net
        self.clip_local_files_only = clip_local_files_only
        self.dino_model_name = dino_model_name
        self.dino_global_patch_size = dino_global_patch_size
        self._lpips_model = None
        self._lpips_spatial_model = None
        self._clip_model = None
        self._clip_processor = None
        self._structure_distance_model = None

    def _lazy_lpips(self):
        if self._lpips_model is not None:
            return self._lpips_model
        self._lpips_model = lpips.LPIPS(net=self.lpips_net).to(self.device).eval()
        return self._lpips_model

    def _lazy_lpips_spatial(self):
        if self._lpips_spatial_model is not None:
            return self._lpips_spatial_model
        self._lpips_spatial_model = lpips.LPIPS(net=self.lpips_net, spatial=True).to(self.device).eval()
        return self._lpips_spatial_model

    def _lazy_clip(self):
        if self._clip_model is not None and self._clip_processor is not None:
            return self._clip_model, self._clip_processor
        clip_config = None
        if self.clip_local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            clip_config = CLIPConfig.from_pretrained(self.clip_model_id, local_files_only=True)
        self._clip_processor = CLIPProcessor.from_pretrained(
            self.clip_model_id,
            local_files_only=self.clip_local_files_only,
        )
        self._clip_model = CLIPModel.from_pretrained(
            self.clip_model_id,
            config=clip_config,
            local_files_only=self.clip_local_files_only,
        ).to(self.device).eval()
        return self._clip_model, self._clip_processor

    def _lazy_structure_distance(self):
        if self._structure_distance_model is not None:
            return self._structure_distance_model
        os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[2] / ".cache" / "torch"))
        self._structure_distance_model = DinoStructureDistanceCalculator(
            model_name=self.dino_model_name,
            patch_size=self.dino_global_patch_size,
            device=self.device,
        ).eval()
        return self._structure_distance_model

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.dtype == np.uint8:
            return array
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return np.uint8(np.clip(array, 0, 255))

    @staticmethod
    def _normalize_prompt_text(text: str) -> str:
        return " ".join(str(text).strip().split())

    @staticmethod
    def _broadcast_mask(mask: np.ndarray | None, image: np.ndarray) -> np.ndarray | None:
        if mask is None:
            return None
        mask_arr = np.asarray(mask, dtype=np.float32)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        if mask_arr.max(initial=0) > 1.0:
            mask_arr = mask_arr / 255.0
        if mask_arr.shape[:2] != image.shape[:2]:
            mask_image = Image.fromarray(np.uint8(np.clip(mask_arr, 0.0, 1.0) * 255.0))
            mask_image = mask_image.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
            mask_arr = np.asarray(mask_image, dtype=np.float32) / 255.0
        return mask_arr[:, :, None]

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        image_arr = self._to_uint8(image).astype(np.float32)
        mask_arr = self._broadcast_mask(mask, image_arr)
        if mask_arr is None:
            return image_arr
        return image_arr * mask_arr

    @staticmethod
    def _to_lpips_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image.astype(np.float32))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def compute_lpips(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_pred: np.ndarray | None = None,
    ) -> float:
        model = self._lazy_lpips()
        ref = self._apply_mask(reference, mask_ref) / 255.0
        pred = self._apply_mask(prediction, mask_pred) / 255.0
        ref_tensor = self._to_lpips_tensor(ref).to(self.device)
        pred_tensor = self._to_lpips_tensor(pred).to(self.device)
        with torch.no_grad():
            return float(model(pred_tensor, ref_tensor).item())

    def compute_psnr(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_pred: np.ndarray | None = None,
    ) -> float:
        ref = self._apply_mask(reference, mask_ref) / 255.0
        pred = self._apply_mask(prediction, mask_pred) / 255.0
        return float(peak_signal_noise_ratio(ref, pred, data_range=1.0))

    def compute_mse(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_pred: np.ndarray | None = None,
    ) -> float:
        ref = self._apply_mask(reference, mask_ref) / 255.0
        pred = self._apply_mask(prediction, mask_pred) / 255.0
        return float(np.mean((pred - ref) ** 2))

    def compute_ssim(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_pred: np.ndarray | None = None,
    ) -> float:
        ref = self._apply_mask(reference, mask_ref) / 255.0
        pred = self._apply_mask(prediction, mask_pred) / 255.0
        return float(structural_similarity(ref, pred, data_range=1.0, channel_axis=2))

    def compute_clip_score(self, image: np.ndarray, text: str, mask: np.ndarray | None = None) -> float | None:
        normalized_text = self._normalize_prompt_text(text)
        if not normalized_text:
            return None
        model, processor = self._lazy_clip()
        masked = np.uint8(np.clip(self._apply_mask(image, mask), 0.0, 255.0))
        inputs = processor(
            text=[normalized_text],
            images=[Image.fromarray(masked)],
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float(max(100.0 * (image_embeds * text_embeds).sum(dim=-1).item(), 0.0))

    def compute_structure_distance(
        self,
        reference: np.ndarray,
        prediction: np.ndarray,
        mask_ref: np.ndarray | None = None,
        mask_pred: np.ndarray | None = None,
    ) -> float:
        model = self._lazy_structure_distance()
        ref = self._apply_mask(reference, mask_ref)
        pred = self._apply_mask(prediction, mask_pred)
        ref_tensor = torch.from_numpy(np.transpose(ref, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        pred_tensor = torch.from_numpy(np.transpose(pred, axes=(2, 0, 1))).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = model.calculate_global_ssim_loss(ref_tensor, pred_tensor)
        return float(score.detach().cpu().item())

    def compute_locality_ratio(self, source_image: np.ndarray, edited_image: np.ndarray, mask: np.ndarray) -> float | None:
        model = self._lazy_lpips_spatial()
        source = self._apply_mask(source_image, None) / 255.0
        edited = self._apply_mask(edited_image, None) / 255.0
        source_tensor = self._to_lpips_tensor(source).to(self.device)
        edited_tensor = self._to_lpips_tensor(edited).to(self.device)
        with torch.no_grad():
            spatial = model(source_tensor, edited_tensor)
        delta = spatial.squeeze()
        if delta.ndim == 0:
            return None
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).to(self.device)
        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor[..., 0]
        if mask_tensor.max().item() > 1.0:
            mask_tensor = mask_tensor / 255.0
        if tuple(mask_tensor.shape) != tuple(delta.shape[-2:]):
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=delta.shape[-2:],
                mode="nearest",
            ).squeeze()
        change_in = float((delta * mask_tensor).sum().item())
        change_out = float((delta * (1.0 - mask_tensor)).sum().item())
        denom = change_in + change_out
        if denom < 1e-8:
            return None
        return float(change_in / denom)

    def compute_all_metrics(
        self,
        source_image: np.ndarray,
        edited_image: np.ndarray,
        target_prompt: str | None = None,
        source_reconstruction: np.ndarray | None = None,
        *,
        source_prompt: str | None = None,
        target_reference: np.ndarray | None = None,
        edit_mask: np.ndarray | None = None,
        enable_lpips: bool = True,
        enable_psnr: bool = True,
        enable_mse: bool = True,
        enable_ssim: bool = True,
        enable_clip_score: bool = True,
        enable_structure_distance: bool = False,
        enable_locality_ratio: bool = True,
    ) -> dict:
        metrics: dict[str, float | str | None] = {
            "edit_reference_mode": "target_reference" if target_reference is not None else "missing",
        }

        if source_reconstruction is not None:
            if enable_psnr:
                self._safe(metrics, "source_recon_psnr", self.compute_psnr, source_image, source_reconstruction)
            if enable_lpips:
                self._safe(metrics, "source_recon_lpips", self.compute_lpips, source_image, source_reconstruction)

        if enable_clip_score:
            if source_prompt:
                self._safe(metrics, "clip_similarity_source_image", self.compute_clip_score, source_image, source_prompt)
            if target_prompt:
                self._safe(metrics, "clip_similarity_target_image", self.compute_clip_score, edited_image, target_prompt)
                if edit_mask is not None:
                    self._safe(
                        metrics,
                        "clip_similarity_target_image_edit_part",
                        self.compute_clip_score,
                        edited_image,
                        target_prompt,
                        edit_mask,
                    )

        if enable_psnr:
            self._safe(metrics, "psnr", self.compute_psnr, source_image, edited_image)
        if enable_mse:
            self._safe(metrics, "mse", self.compute_mse, source_image, edited_image)
        if enable_ssim:
            self._safe(metrics, "ssim", self.compute_ssim, source_image, edited_image)
        if enable_lpips:
            self._safe(metrics, "lpips", self.compute_lpips, source_image, edited_image)
        if enable_structure_distance:
            self._safe(metrics, "structure_distance", self.compute_structure_distance, source_image, edited_image)

        if target_reference is not None:
            if enable_psnr:
                self._safe(metrics, "edit_ref_psnr", self.compute_psnr, target_reference, edited_image)
            if enable_lpips:
                self._safe(metrics, "edit_ref_lpips", self.compute_lpips, target_reference, edited_image)

        if edit_mask is not None:
            unedit_mask = 1.0 - np.asarray(edit_mask, dtype=np.float32)
            if enable_psnr:
                self._safe(metrics, "psnr_unedit_part", self.compute_psnr, source_image, edited_image, unedit_mask, unedit_mask)
            if enable_mse:
                self._safe(metrics, "mse_unedit_part", self.compute_mse, source_image, edited_image, unedit_mask, unedit_mask)
            if enable_ssim:
                self._safe(metrics, "ssim_unedit_part", self.compute_ssim, source_image, edited_image, unedit_mask, unedit_mask)
            if enable_lpips:
                self._safe(metrics, "lpips_unedit_part", self.compute_lpips, source_image, edited_image, unedit_mask, unedit_mask)
            if enable_structure_distance:
                self._safe(
                    metrics,
                    "structure_distance_unedit_part",
                    self.compute_structure_distance,
                    source_image,
                    edited_image,
                    unedit_mask,
                    unedit_mask,
                )
            if enable_lpips and enable_locality_ratio:
                self._safe(metrics, "locality_ratio", self.compute_locality_ratio, source_image, edited_image, edit_mask)

        metrics["clip_similarity"] = metrics.get("clip_similarity_target_image")
        metrics["clip_score"] = metrics.get("clip_similarity_target_image")
        metrics["clip_similarity_edit_part"] = metrics.get("clip_similarity_target_image_edit_part")
        metrics["clip_score_edit_part"] = metrics.get("clip_similarity_target_image_edit_part")
        metrics["edit_source_psnr"] = metrics.get("psnr")
        metrics["edit_source_lpips"] = metrics.get("lpips")
        metrics["outside_mse"] = metrics.get("mse_unedit_part")
        metrics["outside_psnr"] = metrics.get("psnr_unedit_part")
        metrics["outside_ssim"] = metrics.get("ssim_unedit_part")
        metrics["outside_lpips"] = metrics.get("lpips_unedit_part")
        return metrics

    @staticmethod
    def _safe(metrics: dict, key: str, fn, *args) -> None:
        try:
            metrics[key] = fn(*args)
        except Exception as exc:
            metrics[key] = None
            metrics[f"{key}_error"] = str(exc)
