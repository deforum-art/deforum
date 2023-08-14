"""
This module initializes with configurations provided by `DeforumConfig` class. It loads the 
relevant model (ie `SDLoader`, `SDXLLoader`) and pipeline (ie `BasePipeline`, `TwoStagePipeline`) 
based on the configuration. Later the model and pipeline can be switched on-demand using provided 
methods. Generation with set configurations is performed using the `generate` method within 
the `Deforum` class.

Classes
----------
DeforumConfig : Constructs all the necessary configurations for the Deforum object.
validate_xformers : Validation function to check the availability of xformers.
validate_set_use_flash_attn_2 : Validation function to check the availability of flash_attn_2.

Exceptions
----------
Raises ValueError when an unknown model or pipeline type is provided in the configuration.
"""
from typing import Literal, Optional
import torch
from diffusers.utils import is_xformers_available
from pydantic import validator
from deforum.utils import MixedModel
from . import DefaultBase

# Default model name and device settings
MODEL_NAME_DEFAULT = "Lykon/AbsoluteReality"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Guides for installation of necessary packages
XFORMERS_INSTALLATION_GUIDE = "https://github.com/facebookresearch/xformers"
FLASH_ATTN_2_INSTALLATION_GUIDE = "https://github.com/Dao-AILab/flash-attention"


class DeforumConfig(DefaultBase):
    """Configurations for the Deforum model.

    Parameters
    ----------
    model_name : str, optional
        Name of the model to be used, by default `MODEL_NAME_DEFAULT`.
    dtype : torch.dtype, optional
        Type of the data to be used, by default `torch.float16`.
    device : torch.device
        Device to be used for computations (CPU or CUDA), by default `DEVICE`.
    variant : str, optional
        Variant of the DeForUM model.
    use_safetensors : bool, optional
        Flag indicating whether to use safe tensors.

    model_type : str, optional
        Type of the model to be used, can be "sdxl", "sd1.5", "sd2.1".
        Default is "sd1.5".
    pipeline_type : str, optional
        Type of the pipeline to be used, can be "base", "vid2vid", "2stage".
        Default is "base".

    use_xformers : bool, optional
        Flag indicating whether to use the xformers library.
        Requires the xformers library to be installed before use, by default is False.
    set_use_flash_attn_2 : bool, optional
        Flag indicating whether to use the flash_attn_2 library.
        Requires the flash_attn_2 library to be installed, by default is False.
    mixed_model : MixedModel, optional
        Instance of the mixed model to be used, by default is None.
    """

    model_name: str = MODEL_NAME_DEFAULT
    dtype: torch.dtype = torch.float16
    device: torch.device = DEVICE
    variant: Optional[str]
    use_safetensors: Optional[bool]

    model_type: Literal["sdxl", "sd1.5", "sd2.1"] = "sd1.5"
    pipeline_type: Optional[Literal["base", "vid2vid", "2stage"]] = "base"

    use_xformers: Optional[bool] = False
    set_use_flash_attn_2: Optional[bool] = False
    mixed_model: MixedModel = None

    @validator("use_xformers", pre=True)
    def validate_xformers(cls, value, values, config, field):
        """Validation function to check the availability of xformers.

        Parameters
        ----------
        value : bool
            The provided value for `use_xformers`.

        Raises
        ------
        ImportError
            If `use_xformers` is True but the xformers library is not installed.
        """
        if value and not is_xformers_available():
            raise ImportError(
                f"xformers is not installed. Please install it \
                following the instructions on {XFORMERS_INSTALLATION_GUIDE}"
            )
        return value

    @validator("set_use_flash_attn_2", pre=True)
    def validate_set_use_flash_attn_2(cls, value, values, config, field):
        """Validation function to check the availability of flash_attn_2.

        Parameters
        ----------
        value : bool
            The provided value for `set_use_flash_attn_2`.

        Raises
        ------
        ImportError
            If `set_use_flash_attn_2` is True but the flash_attn_2 library is not installed.
        """
        if value:
            try:
                import flash_attn_2_cuda
            except ImportError as exc:
                raise ImportError(
                    f"flash_attn_2_cuda is not installed. Please \
                    install it following the instructions on {FLASH_ATTN_2_INSTALLATION_GUIDE}"
                ) from exc
        return value
