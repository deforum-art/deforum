from typing import Union
from pydantic import BaseConfig

import torch

from deforum.backend.models import AbstractPipeline
from deforum.typed_classes import GenerationArgs, ResultBase
from deforum.utils.image_utils import ImageHandler
from deforum.utils.helpers import parse_seed_for_mode


class BasePipeline:
    args_type = GenerationArgs

    def __init__(self, model: AbstractPipeline, config: BaseConfig) -> None:
        """
        initialization hook for the pipeline.
        For use when the pipeline needs to load something such as a depth model, raft optical flow model, etc.
        """
        pass

    def sample(
        self,
        model: AbstractPipeline,
        args: GenerationArgs,
    ) -> ResultBase:
        """
        Generate a sample image from the given text prompt.
        """
        images = []
        if args.seed_mode == "ladder":
            if args.seed_list is None:
                args.seed_list = [args.seed, args.seed + 1]

        for _ in range(args.repeat):
            seed = parse_seed_for_mode(args.seed, args.seed_mode, args.seed_list)
            model.scheduler = args.sampler.to_scheduler().from_config(model.scheduler.config)
            args.generator = torch.Generator(model.device).manual_seed(seed)
            images_ = model(
                **args.to_kwargs(),
                output_type="pt",
            ).images
            if args.save_intermediates:
                ImageHandler.save_images(
                    ResultBase(
                        image=images_,
                        output_type="pt",
                        args=args,
                    ),
                    template_str=args.template_save_path,
                    image_index=-1,
                )

            images.append(images_.cpu())

        images = torch.cat(images, 0)
        return ResultBase(
            image=images,
            output_type="pt",
            args=args,
        )
