from pydantic import BaseConfig
import torch

from deforum.backend import AbstractPipeline
from deforum.typed_classes import ResultBase
from deforum.typed_classes.generation_args_two_stage import GenerationArgsTwoStage
from deforum.utils.helpers import parse_seed_for_mode
from deforum.utils.image_utils import resize_tensor_result
from .base_pipeline import BasePipeline


class TwoStagePipeline:
    args_type = GenerationArgsTwoStage

    def __init__(self, model: AbstractPipeline, config: BaseConfig) -> None:
        self.base_pipeline = BasePipeline(model, config)

    def sample(
        self,
        model: AbstractPipeline,
        args: GenerationArgsTwoStage,
    ) -> ResultBase:
        """
        Generate a sample image from the given text prompt with two stages.
        """

        images = []

        args = self.base_pipeline.prep_seed(args)

        args_for_base = args.copy(exclude={"repeat", "generator"}, deep=True)

        # Set the repeat to 1 for first stage to avoid unnecessary repetitions.
        args_for_base.repeat = 1
        # Set the return_images to True to get the images from the first stage
        # which we will use in the second stage.
        args_for_base.return_images = True

        for _ in range(args.repeat):
            ## Stage 1 of the two-stage pipeline
            args_cpy = args_for_base.copy()
            args_cpy.save_intermediates = False

            args.seed = parse_seed_for_mode(args.seed, args.seed_mode, args.seed_list)
            args_cpy.seed = args.seed
            args_cpy.seed_list = args.seed_list
            args_cpy.seed_mode = "constant"

            # Run inference to generate the first images
            stage1_result = self.base_pipeline.sample(model, args_cpy)

            ## Stage 2 of the two-stage pipeline

            # Resize the images to the result 2x size
            stage1_result = resize_tensor_result(stage1_result, (args.height * 2, args.width * 2), tensor_format="nchw")

            stage2_args = stage1_result.args.copy(deep=True, exclude={"generator"})

            # Update stage2_args with values from the GenerationArgsTwoStage object
            stage2_args.save_intermediates = args.save_intermediates
            stage2_args.return_images = args.return_images
            stage2_args.prompt = args.prompt_stage2 or args.prompt
            stage2_args.eta = args.eta_stage2 or args.eta
            stage2_args.num_inference_steps = args.num_inference_steps_stage2 or args.num_inference_steps
            stage2_args.sampler = args.sampler_stage2 or args.sampler
            stage2_args.guidance_scale = args.guidance_stage2 or args.guidance_scale
            stage2_args.strength = args.strength_stage2 or args.strength
            stage2_args.negative_prompt = args.negative_prompt_stage2 or args.negative_prompt

            # Run inference to generate the final images
            stage2_result = self.base_pipeline.sample(model, stage2_args)

            # Save the images if the args dictate so
            if args.return_images:
                images.append(stage2_result.image.cpu())

        # Concatenate the results if the args dictate so, otherwise images will be None
        if args.return_images:
            images = torch.cat(images, 0)
            stage2_result.image = images
        else:
            stage2_args.image = None

        return stage2_args
