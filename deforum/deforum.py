"""
This module provides the Deforum class which is used for image and video generation.
"""
import os

from loguru import logger
import torch
from deforum.backend.custom_text_encoder import CustomCLIPTextModel
from deforum.typed_classes import DeforumConfig, GenerationArgs
from deforum.backend import SDLPWPipelineOneFive
from deforum.utils.pytorch_optimizations import channels_last
from deforum.utils.image_utils import ImageReader

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
            self.pipe = SDLPWPipelineOneFive.from_pretrained(
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
        
        if config.unet_channels_last:
            self.pipe = channels_last(self.pipe)

        self.pipe.to(self.device)

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

    def sample(self, args: GenerationArgs):
        """
        Generate a sample image from the given text prompt.
        """

        if args.seed is not None and args.generator is None:
            args.generator = torch.Generator(device=self.device).manual_seed(args.seed)

        images = self.pipe(**args.dict(exclude={"output_type",'seed'}),output_type='pt').images
        for i in range(len(images)):
            ImageReader.write_image_torch(
                images[i],
                os.path.join(self.samples_dir, self.sample_format.format(i)),
            )