from pydantic import BaseConfig

from ..backend import AbstractPipeline
from ..typed_classes import GenerationArgsAnimation, ResultBase
from .base_pipeline import BasePipeline


class DeforumPipeline(BasePipeline):
    args_type = GenerationArgsAnimation
    def __init__(self, model: AbstractPipeline, config: BaseConfig) -> None:
        self.base_pipeline = BasePipeline(model, config)

    def sample(
        self,
        model: AbstractPipeline,
        args: GenerationArgsAnimation,
    ) -> ResultBase:
        """
        Generate a sample image from the given text prompt.
        """
        raise NotImplementedError("DeforumPipeline is not implemented yet.")
