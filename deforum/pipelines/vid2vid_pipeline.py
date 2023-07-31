from ..backend.models import AbstractPipeline
from ..typed_classes import GenerationVideo, ResultBase
from .base_pipeline import BasePipeline


class Vid2VidPipeline(BasePipeline):
    args_type = GenerationVideo

    def sample(
        self,
        model: AbstractPipeline,
        args: GenerationVideo,
    ) -> ResultBase:
        """
        Generate a sample image from the given text prompt.
        """
        raise NotImplementedError("Vid2VidPipeline is not implemented yet.")
