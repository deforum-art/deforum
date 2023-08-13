import torch
from loguru import logger
from deforum.typed_classes import DeforumConfig
from deforum.backend import StableDiffusionXLPipeline, AttnProcessorFlash2_2_0
from deforum.utils import enable_optimizations, channels_last


class SDXLLoader:

    @staticmethod
    def _load_pipe(config: DeforumConfig, extra: dict) -> StableDiffusionXLPipeline:

        common_kwargs = {
            "torch_dtype": config.dtype,
            "variant": config.variant,
            "use_safetensors": config.use_safetensors,
        }

        return StableDiffusionXLPipeline.from_pretrained(
            config.model_name,
            **common_kwargs,
            **extra,
        )

    @classmethod
    def load(cls, config: DeforumConfig) -> StableDiffusionXLPipeline:

        enable_optimizations()
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
        pipe.vae.to(memory_format=torch.contiguous_format)
        pipe = channels_last(pipe)

        return pipe

