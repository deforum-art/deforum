from pydantic import BaseConfig
import torch

from deforum.backend.models import AbstractPipeline
from deforum.typed_classes import ResultBase
from deforum.typed_classes.generation_args_two_stage import GenerationArgsTwoStage
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
        args_for_base = args.copy(exclude={"repeat", "generator"})
        save_intermediates = args.save_intermediates
        args_for_base.repeat = 1
        for _ in range(args.repeat):
            args_cpy = args_for_base.copy(deep=True)
            args_cpy.save_intermediates = False
            stage1_result = self.base_pipeline.sample(model, args_cpy)
            stage1_result = resize_tensor_result(stage1_result, (args.height * 2, args.width * 2), tensor_format="nchw")
            stage2_args = stage1_result.args.copy(deep=True, exclude={"generator"})
            stage2_args.save_intermediates = save_intermediates

            # Update stage2_args with values from the GenerationArgsTwoStage object

            stage2_args.prompt = args.prompt_stage2 or args.prompt
            stage2_args.eta = args.eta_stage2 or args.eta
            stage2_args.num_inference_steps = args.num_inference_steps_stage2 or args.num_inference_steps
            stage2_args.sampler = args.sampler_stage2 or args.sampler
            stage2_args.guidance_scale = args.guidance_stage2 or args.guidance_scale
            stage2_args.strength = args.strength_stage2 or args.strength
            stage2_args.negative_prompt = args.negative_prompt_stage2 or args.negative_prompt

            stage2_result = self.base_pipeline.sample(model, stage2_args)
            images.append(stage2_result.image.cpu())

        images = torch.cat(images, 0)
        stage2_result.image = images
        return stage2_args
