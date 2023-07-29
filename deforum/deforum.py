"""
This module provides the Deforum class which is used for image and video generation.
"""
import os

from loguru import logger
import torch
from deforum.backend.custom_text_encoder import CustomCLIPTextModel
from deforum.modules.custom_attn_processors.attn_processor_flash_2 import AttnProcessorFlash2_2_0
from deforum.typed_classes import DeforumConfig, GenerationArgs
from deforum.backend import SDLPWPipelineOneFive
from deforum.typed_classes.result_base import ResultBase
from deforum.utils.pytorch_optimizations import channels_last
from deforum.utils.image_utils import ImageHandler


class Deforum:
    """
    Main application class which aids in image or video generation.
    """

    def __init__(self, config: DeforumConfig):
        self.model_name = config.model_name
        self.dtype = config.dtype
        self.variant = config.variant
        self.use_safetensors = config.use_safetensors
        self.device = config.device
        self.samples_dir = config.samples_dir
        self.sample_format = config.sample_format

        try:
            self.pipe: SDLPWPipelineOneFive = SDLPWPipelineOneFive.from_pretrained(
                self.model_name,
                text_encoder=CustomCLIPTextModel.from_pretrained(
                    self.model_name,
                    subfolder="text_encoder",
                    torch_dtype=config.dtype,
                    variant=self.variant,
                    use_safetensors=config.use_safetensors,
                ),
                torch_dtype=self.dtype,
                variant=self.variant,
                use_safetensors=self.use_safetensors,
            )
        except Exception as e:
            logger.exception(f"Failed to load model! {e}", exc_info=True, stack_info=True)
            exit(0)

        if config.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()
        elif config.set_use_flash_attn_2:
            self.pipe.unet.set_attn_processor(AttnProcessorFlash2_2_0())
            self.pipe.vae.set_attn_processor(AttnProcessorFlash2_2_0())
        if config.unet_channels_last:
            self.pipe = channels_last(self.pipe)

        self.pipe.to(self.device)

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

    def sample(self, args: GenerationArgs):
        """
        Generate a sample image from the given text prompt.
        """

        images = self.pipe(**args.to_kwargs(), output_type="pt").images
        return ResultBase(
            image=images,
            output_type="pt",
            args=args,
        )
