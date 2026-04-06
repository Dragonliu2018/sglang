"""LongCat-Image pipeline for SGLang."""

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image import (
    LongCatImageBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


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
        # 1. Pre-processing: text encoding, latent preparation, timestep setup
        self.add_stage(
            LongCatImageBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                text_processor=self.get_module("text_processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "longcat_image_before_denoising_stage",
        )

        # 2. Standard denoising loop
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 3. Standard VAE decoding
        self.add_standard_decoding_stage()


EntryClass = [LongCatImagePipeline]
