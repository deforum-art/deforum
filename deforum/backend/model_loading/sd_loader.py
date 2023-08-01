"""
This module loads the custom CLIP Text model and SDLPWPipeline (1.5) based on a given 
configuration.

This module uses the loguru, deforum.backend, deforum.modules, deforum.typed_classes 
and deforum.utils.

Raises
------
ValueError: If model fails to load.
"""

from loguru import logger
from deforum.backend import CustomCLIPTextModel, SDLPWPipelineOneFive
from deforum.modules import AttnProcessorFlash2_2_0
from deforum.typed_classes import DeforumConfig
from deforum.utils import channels_last

class SDLoader:
    """
    A class used to load the SDLPWPipeline (1.5) model.

    Methods
    -------
    _load_pipe(config, extra)
        Returns the SDLPWPipeline (1.5) instantiated with given configuration.
    load(config)
        Returns the SDLPWPipeline (1.5) loaded with the given configuration and 
        potentially exceptions handled.
    """

    @staticmethod
    def _load_pipe(config: DeforumConfig, extra: dict) -> SDLPWPipelineOneFive:
        """
        A static method that loads the SDLPWPipeline (1.5) given the configuration 
        and extra instructions.
        
        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.
        extra : dict
            Dictionary containing additional instructions.

        Returns
        -------
        SDLPWPipelineOneFive
            The SDLPWPipeline (1.5) loaded with the given parameters.
        """
        common_kwargs = {"torch_dtype": config.dtype, "variant": config.variant, \
                         "use_safetensors": config.use_safetensors}

        text_encoder = CustomCLIPTextModel.from_pretrained(
            config.model_name,
            subfolder="text_encoder",
            **common_kwargs
        )

        return SDLPWPipelineOneFive.from_pretrained(
            config.model_name,
            text_encoder=text_encoder,
            **common_kwargs,
            **extra,
        )

    @classmethod
    def load(cls, config: DeforumConfig) -> SDLPWPipelineOneFive:
        """
        Method that handles exceptions during the model loading process and loads 
        the SDLPWPipeline (1.5).
        
        Parameters
        ----------
        config : DeforumConfig
            The configurations for Deforum.

        Returns
        -------
        SDLPWPipelineOneFive
            The SDLPWPipeline (1.5) loaded with the given parameters, exceptions handled.

        Raises
        ------
        ValueError
            If model fails to load.
        """
        extra = {}
        if config.mixed_model:
            extra["unet"] = config.mixed_model.merge().to(dtype=config.dtype)

        try:
            pipe = cls._load_pipe(config, extra)
        except Exception as err:
            logger.exception(f"Failed to load model: {err}", exc_info=True, stack_info=True)
            raise ValueError("Failed to load the model") from err

        if config.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        elif config.set_use_flash_attn_2:
            pipe.unet.set_attn_processor(AttnProcessorFlash2_2_0())
            pipe.vae.set_attn_processor(AttnProcessorFlash2_2_0())
        pipe = channels_last(pipe)
        return pipe
