from typing import Literal, Optional

import torch
from diffusers.utils import is_xformers_available
from pydantic import validator


from deforum.utils import MixedModel
from . import DefaultBase


class DeforumConfig(DefaultBase):
    model_name: str = "Lykon/AbsoluteReality"
    dtype: torch.dtype = torch.float16
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant: Optional[str]
    use_safetensors: Optional[bool]

    model_type: Literal["sdxl", "sd1.5", "sd2.1"] = "sd1.5"
    pipeline_type: Optional[Literal["base", "vid2vid", "2stage"]] = "base"

    use_xformers: Optional[bool] = False
    set_use_flash_attn_2: Optional[bool] = False
    mixed_model: MixedModel = None

    @validator("use_xformers", pre=True)
    def validate_xformers(cls, value, values, config, field):
        if value:
            if not is_xformers_available():
                raise ImportError(
                    "xformers is not installed. Please install it following the instructions on 'https://github.com/facebookresearch/xformers'"
                )
            else:
                return True
        else:
            return False

    @validator("set_use_flash_attn_2", pre=True)
    def validate_set_use_flash_attn_2(cls, value, values, config, field):
        if value:
            try:
                import flash_attn_2_cuda

                return True
            except ImportError:
                raise ImportError(
                    "flash_attn_2_cuda is not installed. Please install it following the instructions on 'https://github.com/Dao-AILab/flash-attention'"
                )
        else:
            return False
