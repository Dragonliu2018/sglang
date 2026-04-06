from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.longcat_image import (
    LongCatImageDitConfig,
)
from sglang.multimodal_gen.configs.models.vaes.longcat_image import (
    LongCatImageVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def _unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    # VAE applies 8x compression, plus 2x packing factor
    h = 2 * (int(height) // (vae_scale_factor * 2))
    w = 2 * (int(width) // (vae_scale_factor * 2))

    latents = latents.view(batch_size, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), h, w)
    return latents


@dataclass
class LongCatImagePipelineConfig(ImagePipelineConfig):
    """Configuration for the LongCat-Image T2I pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_precision: str = "bf16"
    should_use_guidance: bool = True
    vae_tiling: bool = False
    vae_sp: bool = False
    enable_autocast: bool = False

    dit_config: DiTConfig = field(default_factory=LongCatImageDitConfig)
    vae_config: VAEConfig = field(default_factory=LongCatImageVAEConfig)

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "txt_ids": batch.txt_ids,
            "img_ids": batch.img_ids,
            "image_rotary_emb": batch.image_rotary_emb,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "txt_ids": batch.negative_txt_ids,
            "img_ids": batch.img_ids,
            "image_rotary_emb": batch.image_rotary_emb,
        }

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # LongCat uses standard AutoencoderKL: latents = (latents / scaling_factor) + shift_factor
        # DecodingStage does: latents = latents / scaling_factor + shift_factor
        # So return (scaling_factor, shift_factor) directly.
        vae_config = getattr(vae, "config", None)
        if vae_config is not None:
            sf = getattr(vae_config, "scaling_factor", 1.0)
            shift = getattr(vae_config, "shift_factor", 0.0)
        else:
            sf = 1.0
            shift = 0.0
        return sf, shift

    def post_denoising_loop(self, latents, batch):
        vae_scale_factor = self.vae_config.get_vae_scale_factor()
        latents = _unpack_latents(latents, batch.height, batch.width, vae_scale_factor)
        # Add frames dimension for DecodingStage compatibility: [B, C, H, W] -> [B, C, 1, H, W]
        latents = latents.unsqueeze(2)
        return latents

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Remove frames dimension before VAE decode: [B, C, 1, H, W] -> [B, C, H, W]."""
        if latents.dim() == 5 and latents.shape[2] == 1:
            latents = latents.squeeze(2)
        return latents

    def postprocess_cfg_noise(
        self,
        batch,
        noise_pred: torch.Tensor,
        noise_pred_cond: torch.Tensor,
    ) -> torch.Tensor:
        enable_cfg_renorm = getattr(batch, "enable_cfg_renorm", True)
        cfg_renorm_min = getattr(batch, "cfg_renorm_min", 0.0)
        if not enable_cfg_renorm:
            return noise_pred
        cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)
        noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        return noise_pred * scale
