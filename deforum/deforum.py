"""
This module provides the Deforum class which is used for image and video generation.
"""
import os
from typing import Tuple

from loguru import logger
import torch
from deforum.backend.custom_text_encoder import CustomCLIPTextModel
from deforum.modules.custom_attn_processors.attn_processor_flash_2 import AttnProcessorFlash2_2_0
from deforum.typed_classes import DeforumConfig, GenerationArgs
from deforum.backend import SDLPWPipelineOneFive
from deforum.typed_classes.result_base import ResultBase
from deforum.utils.pytorch_optimizations import channels_last


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
        extra = {}
        if config.multi_model:
            extra["unet"] = config.multi_model.merge().to(self.device).to(dtype=self.dtype)

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
                **extra,
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

    def sample(self, args: GenerationArgs, two_stage=False, strength=0.21, save_intermediates=False):
        """
        Generate a sample image from the given text prompt.
        """
        self.pipe.scheduler = args.sampler.to_scheduler().from_config(self.pipe.scheduler.config)
        images = []
        for idx in range(args.repeat):
            args_cpy = args.copy(deep=True)
            args_cpy.seed += idx
            images_ = self.pipe(**args_cpy.to_kwargs(), output_type="pt").images
            if two_stage:
                temp_result = self.resize(ResultBase(image=images_, args=args_cpy), (args.height * 2, args.width * 2))
                temp_args = temp_result.args.copy(deep=True, exclude={"generator"})
                temp_args.generator = torch.Generator(self.device).manual_seed(args_cpy.seed)
                temp_args.strength = strength
                self.pipe.scheduler = args.sampler.to_scheduler().from_config(self.pipe.scheduler.config)
                images_ = self.pipe(**temp_args.to_kwargs(), output_type="pt").images
            if save_intermediates:
                ResultBase(
                    image=images_,
                    output_type="pt",
                    args=args,
                ).save_images(custom_index=idx * images_.shape[0])

            images.append(images_.cpu())

        images = torch.cat(images, 0)
        return ResultBase(
            image=images,
            output_type="pt",
            args=args,
        )

    def resize(self, result: ResultBase, new_size: Tuple[int, int]):
        """
        Resize the given image to the given size.

            args (GenerationArgs): The arguments used to generate the image.
            result (ResultBase): The image to resize.
            new_size (Tuple[int,int]): The new size of the image in (height, width) format.

        """
        result.image = torch.nn.functional.interpolate(
            result.image.permute(0, 3, 1, 2), size=new_size, mode="bicubic", antialias=True
        )
        result.args.height = new_size[0]
        result.args.width = new_size[1]
        result.args.image = result.image
        return result
