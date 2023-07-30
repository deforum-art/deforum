"""
This module provides the Deforum class which is used for image and video generation.
"""
import os
from typing import Union

import torch
from loguru import logger

from deforum.backend import SDLoader, SDXLLoader
from deforum.typed_classes import DeforumConfig, GenerationArgs, ResultBase
from deforum.utils import (
    ImageHandler,
    TemplateParser,
    enable_optimizations,
    resize_tensor_result,
)


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
        if config.pipeline_type == "sd1.5" or config.pipeline_type == "sd2.1":
            self.pipe = SDLoader.load(config)
        elif config.pipeline_type == "sdxl":
            self.pipe = SDXLLoader.load(config)
        else:
            raise ValueError(
                f"Unknown pipeline type in config: {config.pipeline_type}, must be one of 'sd1.5', 'sdxl', 'sd2.1'"
            )

        if config.enable_pytorch_optimizations:
            enable_optimizations()

        self.pipe.to(self.device)

        if not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)

    def sample(
        self,
        args: GenerationArgs,
        two_stage=False,
        strength=0.21,
        save_intermediates=False,
        template_save_path: Union[str, TemplateParser] = "samples/$custom_$timestr_$prompt_$index",
    ):
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
                temp_result = resize_tensor_result(
                    ResultBase(image=images_, args=args_cpy), (args.height * 2, args.width * 2)
                )
                temp_args = temp_result.args.copy(deep=True, exclude={"generator"})
                temp_args.generator = torch.Generator(self.device).manual_seed(args_cpy.seed)
                temp_args.strength = strength
                self.pipe.scheduler = args.sampler.to_scheduler().from_config(self.pipe.scheduler.config)
                images_ = self.pipe(**temp_args.to_kwargs(), output_type="pt").images
            if save_intermediates:
                ImageHandler.save_images(
                    ResultBase(
                        image=images_,
                        output_type="pt",
                        args=args,
                    ),
                    template_str=template_save_path,
                    image_index=-1,
                )

            images.append(images_.cpu())

        images = torch.cat(images, 0)
        return ResultBase(
            image=images,
            output_type="pt",
            args=args,
        )
