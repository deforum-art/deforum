from loguru import logger
import torch
from deforum.backend import CustomCLIPTextModel, StableDiffusionPipeline, AttnProcessorFlash2_2_0
from deforum.typed_classes import DeforumConfig
from deforum.utils import channels_last, enable_optimizations
from diffusers import ControlNetModel


class SDLoader:

    @staticmethod
    def _load_controlnet(config: DeforumConfig):
        if isinstance(config.controlnet, str):
            return [ControlNetModel.from_pretrained(config.controlnet, torch_dtype=config.dtype)]
        if isinstance(config.controlnet, (list, tuple)):
            return [ControlNetModel.from_pretrained(cn, torch_dtype=config.dtype) for cn in config.controlnet]

    @staticmethod
    def _load_pipe(cls, config: DeforumConfig, extra: dict) -> StableDiffusionPipeline:

        common_kwargs = {
            "torch_dtype": config.dtype,
            "variant": config.variant,
            "use_safetensors": config.use_safetensors,
        }

        text_encoder = CustomCLIPTextModel.from_pretrained(config.model_name, subfolder="text_encoder", **common_kwargs)
        controlnet = cls._load_controlnet(config) if config.controlnet else None

        return StableDiffusionPipeline.from_pretrained(
            config.model_name,
            text_encoder=text_encoder,
            controlnet=controlnet,
            **common_kwargs,
            **extra,
        )

    @classmethod
    def load(cls, config: DeforumConfig) -> StableDiffusionPipeline:

        enable_optimizations()
        extra = {}
        if config.mixed_model:
            extra["unet"] = config.mixed_model.merge().to(dtype=config.dtype)

        try:
            pipe = cls._load_pipe(cls, config, extra)
        except Exception as err:
            logger.exception(f"Failed to load model: {err}", exc_info=True, stack_info=True)
            raise ValueError("Failed to load the model") from err

        if config.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        elif config.set_use_flash_attn_2:
            pipe.unet.set_attn_processor(AttnProcessorFlash2_2_0())
            pipe.vae.set_attn_processor(AttnProcessorFlash2_2_0())
        pipe.vae.to(memory_format=torch.contiguous_format)
        pipe = channels_last(pipe)

        return pipe