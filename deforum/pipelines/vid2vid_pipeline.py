import torch
from typing import Union
from ..backend.models import AbstractPipeline
from ..typed_classes import GenerationVideo, ResultBase
from .base_pipeline import BasePipeline
from ..utils import TemplateParser, resize_tensor_result, ImageHandler


class Vid2VidPipeline(BasePipeline):
    args_type = GenerationVideo

    def sample(
        self,
        model: AbstractPipeline,
        args: GenerationVideo,
        two_stage=False,
        strength=0.21,
        save_intermediates=False,
        template_save_path: Union[str, TemplateParser] = "samples/$prompt/$timestr/$custom_$index",
    ):
        """
        Generate a sample image from the given text prompt.
        """
        model.scheduler = args.sampler.to_scheduler().from_config(model.scheduler.config)
        images = []
        for idx in range(args.repeat):
            args_cpy = args.copy(deep=True)
            args_cpy.seed += idx
            images_ = model(**args_cpy.to_kwargs(), output_type="pt").images
            if two_stage:
                temp_result = resize_tensor_result(
                    ResultBase(image=images_, args=args_cpy), (args.height * 2, args.width * 2)
                )
                temp_args = temp_result.args.copy(deep=True, exclude={"generator"})
                temp_args.generator = torch.Generator(self.device).manual_seed(args_cpy.seed)
                temp_args.strength = strength
                model.scheduler = args.sampler.to_scheduler().from_config(model.scheduler.config)
                images_ = model(**temp_args.to_kwargs(), output_type="pt").images
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
