from deforum.backend import SDLoader, SDXLLoader
from deforum.typed_classes import DeforumConfig
from deforum.typed_classes.generation_args import GenerationArgs
from deforum.utils import enable_optimizations
from deforum.pipelines import BasePipeline, TwoStagePipeline


class Deforum:
    def __init__(self, config: DeforumConfig):
        self.load_model(config)
        self.load_pipeline(config)

    def load_model(self, config: DeforumConfig):
        if config.model_type in ["sd1.5", "sd2.1"]:
            self.model = SDLoader.load(config)
        elif config.model_type == "sdxl":
            self.model = SDXLLoader.load(config)
        else:
            raise ValueError(
                f"Unknown model type in config: {config.model_type}, must be one of 'sd1.5', 'sdxl', 'sd2.1'"
            )
        enable_optimizations()
        self.model.to(config.device)

    def load_pipeline(self, config: DeforumConfig):
        if config.pipeline_type in ["base"]:
            self.pipeline = BasePipeline(self.model, config)
        elif config.pipeline_type in ["2stage"]:
            self.pipeline = TwoStagePipeline(self.model, config)
        else:
            raise ValueError(f"Unknown pipeline type in config: {config.pipeline_type}, must be 'base'")

    def switch_model(self, config: DeforumConfig):
        # TODO: Check if model is compatible with pipeline
        self.load_model(config)

    def switch_pipeline(self, config: DeforumConfig):
        # TODO: Check if pipeline is compatible with model
        self.load_pipeline(config)

    def generate(self, args: GenerationArgs, *_args, **kwargs):
        assert isinstance(
            args, self.pipeline.args_type
        ), f"Expected args of type {self.pipeline.args_type}, got {type(args)}"
        return self.pipeline.sample(self.model, args, *_args, **kwargs)
