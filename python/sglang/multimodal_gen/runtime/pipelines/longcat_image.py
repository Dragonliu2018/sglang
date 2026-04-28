"""LongCat-Image pipeline for SGLang."""

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image import (
    LongCatImageBeforeDenoisingStage,
    LongCatImageRoPEStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _prepare_mu(batch, server_args):
    """Compute mu for FlowMatchEulerDiscreteScheduler from the packed latent token count."""
    from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
        _calculate_shift,
    )

    image_seq_len = batch.latents.shape[1]
    scheduler_config = server_args.pipeline_config.dit_config  # fallback
    # Prefer scheduler config if available on batch
    sched = getattr(batch, "scheduler", None) or server_args.pipeline_config
    cfg = getattr(sched, "config", {})
    mu = _calculate_shift(
        image_seq_len,
        cfg.get("base_image_seq_len", 256) if hasattr(cfg, "get") else 256,
        cfg.get("max_image_seq_len", 4096) if hasattr(cfg, "get") else 4096,
        cfg.get("base_shift", 0.5) if hasattr(cfg, "get") else 0.5,
        cfg.get("max_shift", 1.15) if hasattr(cfg, "get") else 1.15,
    )
    return "mu", mu


class LongCatImagePipeline(LoRAPipeline, ComposedPipelineBase):
    """Pipeline for LongCat-Image text-to-image generation."""

    pipeline_name = "LongCatImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "text_processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        transformer = self.get_module("transformer")

        # 1. Text encoding + prompt rewriting
        self.add_stage(
            LongCatImageBeforeDenoisingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                text_processor=self.get_module("text_processor"),
                transformer=transformer,
            ),
            "longcat_image_before_denoising_stage",
        )

        # 2. Latent preparation (batch-size-aware via pipeline config hooks)
        self.add_standard_latent_preparation_stage()

        # 3. RoPE pre-computation (needs img_ids from latent stage)
        self.add_stage(
            LongCatImageRoPEStage(transformer=transformer),
            "longcat_image_rope_stage",
        )

        # 4. Timestep preparation (mu computed from packed latent token count)
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_prepare_mu],
        )

        # 5. Standard denoising loop
        self.add_standard_denoising_stage()

        # 6. Standard VAE decoding
        self.add_standard_decoding_stage()


EntryClass = [LongCatImagePipeline]
