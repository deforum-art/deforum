"""
This module contains the BasePipeline class which is the parent class for all
pipelines used in Deforum. Deforum uses pipelines to handle the generation, processing,
and saving of images. This includes preparation of seeds, generating images, saving
images, and concatenating the results.

Classes
-------
BasePipeline
    BasePipeline is the parent class for all pipelines in Deform. It provides 
    methods to handle the various stages in the image processing pipeline.

Dependencies
------------
- pydantic: BaseConfig
- torch
- deforum.backend.models: AbstractPipeline
- deforum.typed_classes: GenerationArgs, ResultBase
- deforum.utils.image_utils: ImageHandler
- deforum.utils.helpers: parse_seed_for_mode
"""

from pydantic import BaseConfig
import torch
from deforum.backend.models import AbstractPipeline
from deforum.typed_classes import GenerationArgs, ResultBase
from deforum.utils.image_utils import ImageHandler
from deforum.utils.helpers import parse_seed_for_mode

class BasePipeline:
    """
    BasePipeline is the parent class for all pipelines. 

    Deforum uses pipelines to handle the generation, processing, and saving of images. 

    Attributes
    ----------
    args_type : GenerationArgs
        The argument type that should be used for the generation process.

    Methods
    -------
    __init__(self, model, config)
        Constructs a BasePipeline object.

    prep_seed(args)
        Prepares seed inputs for the generation process.

    generate_images(model, args, seed)
        Handles the image generation process using the provided model, arguments, and seed.

    save_images(args, images_)
        Saves intermediate images if the args dictate so.

    sample(model, args)
        Handles the sampling process. This includes preparing seeds, generating images,
        saving images, and concatenating the results..
    """

    args_type = GenerationArgs

    def __init__(self, model: AbstractPipeline, config: BaseConfig) -> None:
        """
        Constructs the necessary attributes for the BasePipeline object.
        
        Parameters
        ----------
        model : AbstractPipeline
            Model object with pre-defined processing methods.
        config : BaseConfig
            Configuration settings for the pipeline.
        """
        pass

    @staticmethod
    def prep_seed(args):
        """
        Prepares the seeds for the generation process based on the arguments.
        
        Parameters
        ----------
        args :
            Arguments defining how the seeds should be prepared.

        Returns
        -------
        args :
            Arguments with the prepared seeds.
        """
        if args.seed_mode == "ladder" and args.seed_list is None:
            args.seed_list = [args.seed, args.seed + 1]
        return args

    @staticmethod
    def generate_images(model, args, seed):
        """
        Generates images using the provided model, arguments, and seed..
        
        Parameters
        ----------
        model : AbstractPipeline
            Model object with pre-defined processing methods.
        args :
            Parameters for the image generation.
        seed :
            Seed for the random process.

        Returns
        -------
        images_ :
            Generated images.
        """
        model.scheduler = args.sampler.to_scheduler().from_config(model.scheduler.config)
        args.generator = torch.Generator(model.device).manual_seed(seed)
        images_ = model(**args.to_kwargs(), output_type="pt").images
        return images_

    @staticmethod
    def save_images(args, images_):
        """
        Saves intermediate images if dictated by the arguments.
        
        Parameters
        ----------
        args :
            Arguments that dictate whether to save intermediate images or not.
        images_ :
            The intermediate images to be saved.
        """
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

    def sample(self, model: AbstractPipeline, args: GenerationArgs,) -> ResultBase:
        """
        Combines preperation of seeds, generation of images and saving of images 
        into one sampling process.
        
        Parameters
        ----------
        model : AbstractPipeline
            Model object with pre-defined processing methods.
        args : GenerationArgs
            Arguments for the generation process.

        Returns
        -------
        ResultBase
            Result object containing the images, output_type and corresponding arguments.
        """
        images = []
        args = self.prep_seed(args)
        for _ in range(args.repeat):
            seed = parse_seed_for_mode(args.seed, args.seed_mode, args.seed_list)

            images_ = self.generate_images(model, args, seed)
            self.save_images(args, images_)

            images.append(images_.cpu())

        images = torch.cat(images, 0)

        return ResultBase(image=images, output_type="pt", args=args)
