"""
The central class `Deforum` initializes with configurations provided by `DeforumConfig`. 
It loads the relevant model (SDLoader, SDXLLoader) and pipeline (BasePipeline, TwoStagePipeline)
based on configuration. Later the model and pipeline can be switched on-demand using provided 
methods. Generation with set configurations is performed using the `generate` method within 
the `Deforum` class.

Classes:
    Deforum: Main class for operations.

Exceptions:
    Raises ValueError when an unknown model or pipeline type is provided in the configuration.
"""
from deforum.backend import SDLoader, SDXLLoader
from deforum.typed_classes import DeforumConfig
from deforum.typed_classes.generation_args import GenerationArgs
from deforum.utils import enable_optimizations
from deforum.pipelines import BasePipeline, TwoStagePipeline


class Deforum:
    """
    Main class that constructs and operates the Deforum model.

    Attributes
    ----------
    model : type
        The SDLoader or SDXLLoader model used for Deforum.
    pipeline : type
        The pipeline type used in the model (Base or Two-Stage).
    """

    def __init__(self, config: DeforumConfig):
        """
        Constructs all the necessary attributes for the Deforum object.

        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.
        """
        self.load_model(config)
        self.load_pipeline(config)

    def load_model(self, config: DeforumConfig):
        """
        Loads the desired model based on the configuration.

        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.

        Raises
        ------
        ValueError
            If the model type provided in the config is unknown.
        """
        if config.model_type in ["sd1.5", "sd2.1"]:
            self.model = SDLoader.load(config)
        elif config.model_type == "sdxl":
            self.model = SDXLLoader.load(config)
        else:
            raise ValueError(
                f"Unknown model type in config: {config.model_type}, \
                    must be one of 'sd1.5', 'sdxl', 'sd2.1'"
            )
        enable_optimizations()
        self.model.to(config.device)

    def load_pipeline(self, config: DeforumConfig):
        """
        Loads the desired pipeline based on the configuration.

        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.

        Raises
        ------
        ValueError
            If the pipeline type provided in the config is unknown.
        """
        if config.pipeline_type in ["base"]:
            self.pipeline = BasePipeline(self.model, config)
        elif config.pipeline_type in ["2stage"]:
            self.pipeline = TwoStagePipeline(self.model, config)
        else:
            raise ValueError(
                f"Unknown pipeline type in config: \
                             {config.pipeline_type}, must be 'base'"
            )

    def switch_model(self, config: DeforumConfig):
        """
        Switches the current model with a new model based on the configuration.
        Note: Consider checking if the new model is compatible with the pipeline.

        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.
        """
        # TODO: Check if model is compatible with pipeline
        self.load_model(config)

    def switch_pipeline(self, config: DeforumConfig):
        """
        Switches the current pipeline with a new pipeline based on the configuration.
        Note: Consider checking if the new pipeline is compatible with the model.

        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.
        """
        # TODO: Check if pipeline is compatible with model
        self.load_pipeline(config)

    def generate(self, args: GenerationArgs, *_args, **kwargs):
        """
        Generates a sample using the selected model and pipeline.

        Parameters
        ----------
        args : GenerationArgs
            The arguments needed for generation.
        *_args : type
            Extra arguments.
        **kwargs : type
            Extra keyword arguments.

        Returns
        ------
        type
            The generated sample.

        Raises
        ------
        AssertionError
            If the provided arguments are not the same type as the pipeline.
        """
        assert isinstance(
            args, self.pipeline.args_type
        ), f"Expected args of type {self.pipeline.args_type}, got {type(args)}"
        return self.pipeline.sample(self.model, args, *_args, **kwargs)
