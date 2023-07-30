import torch
from loguru import logger

from deforum.backend import CustomCLIPTextModel
from deforum.modules import AttnProcessorFlash2_2_0
from deforum.pipelines import SDLPWPipelineOneFive
from deforum.typed_classes import DeforumConfig
from deforum.utils import channels_last


class SDLoader:
    @classmethod
    def load(cls, config: DeforumConfig) -> SDLPWPipelineOneFive:
        extra = {}
        device = config.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.multi_model:
            extra["unet"] = config.multi_model.merge().to(dtype=config.dtype)

        try:
            pipe: SDLPWPipelineOneFive = SDLPWPipelineOneFive.from_pretrained(
                config.model_name,
                text_encoder=CustomCLIPTextModel.from_pretrained(
                    config.model_name,
                    subfolder="text_encoder",
                    torch_dtype=config.dtype,
                    variant=config.variant,
                    use_safetensors=config.use_safetensors,
                ),
                torch_dtype=config.dtype,
                variant=config.variant,
                use_safetensors=config.use_safetensors,
                **extra,
            )
        except Exception as e:
            logger.exception(f"Failed to load model! {e}", exc_info=True, stack_info=True)
            exit(0)

        if config.use_xformers:
            pipe.enable_xformers_memory_efficient_attention()
        elif config.set_use_flash_attn_2:
            pipe.unet.set_attn_processor(AttnProcessorFlash2_2_0())
            pipe.vae.set_attn_processor(AttnProcessorFlash2_2_0())
        if config.unet_channels_last:
            pipe = channels_last(pipe)
        return pipe
