"""
This module contains the `Deforum` class which wraps a custom text-to-image pipeline.
The class facilitates loading a pretrained model, setting configurations, 
processing user inputs, and generating image outputs.
"""
import os
import sys
from loguru import logger
from deforum.backend.custom_text_encoder import CustomCLIPTextModel
from deforum.modules.custom_attn_processors.attn_processor_flash_2 import AttnProcessorFlash2_2_0
from deforum.typed_classes import DeforumConfig, GenerationArgs
from deforum.backend import SDLPWPipelineOneFive
from deforum.typed_classes.result_base import ResultBase
from deforum.utils.pytorch_optimizations import channels_last

class Deforum:
    """ 
    The `Deforum` class provides a pipeline for generating images based on text inputs. 
    The pipeline uses a pre-trained model and configurations passed during the initialization. 
    It allows custom methods for processing attention, applying transformations, etc.
    """
    def __init__(self, config: DeforumConfig):
        """
        Initialize `Deforum` with the provided configuration.
        
        Args:
            config (DeforumConfig): Configuration for Deforum.
        """
        self.config = config
        self._initialize_pipeline()
        self._apply_configurations()
        self._move_pipeline_to_device()

    def _initialize_pipeline(self):
        """Initialize the pipeline with provided configuration."""
        try:
            self.pipe: SDLPWPipelineOneFive = SDLPWPipelineOneFive.from_pretrained(
                self.config.model_name,
                text_encoder=CustomCLIPTextModel.from_pretrained(
                    self.config.model_name,
                    subfolder="text_encoder",
                    torch_dtype=self.config.dtype,
                    variant=self.config.variant,
                    use_safetensors=self.config.use_safetensors,
                ),
                torch_dtype=self.config.dtype,
                variant=self.config.variant,
                use_safetensors=self.config.use_safetensors,
            )
        except Exception as err:
            logger.exception(f"Failed to load model! {err}", exc_info=True, stack_info=True)
            sys.exit(0)

    def _apply_configurations(self):
        """Apply different attention processors or channel optimizations based on config."""
        if self.config.use_xformers:
            self.pipe.enable_xformers_memory_efficient_attention()
        elif self.config.set_use_flash_attn_2:
            attn_processor = AttnProcessorFlash2_2_0()
            self.pipe.unet.set_attn_processor(attn_processor)
            self.pipe.vae.set_attn_processor(attn_processor)
        if self.config.unet_channels_last:
            self.pipe = channels_last(self.pipe)

    def _move_pipeline_to_device(self):
        """Move the pipeline to the configured device."""
        self.pipe.to(self.config.device)

    def sample(self, args: GenerationArgs):
        """
        Execute the pipeline with the given arguments, generate images and return as ResultBase.

        Args:
            args (GenerationArgs): An instance of the GenerationArgs class.
    
        Returns:
            ResultBase: Image and additional information about the generated result.
        """
        images = self.pipe(**args.to_kwargs(), output_type="pt").images
        return ResultBase(
            image=images,
            output_type="pt",
            args=args,
        )
