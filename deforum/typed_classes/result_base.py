"""
This module defines a class ResultBase that handles operations related to image objects.
Operations include checking output types, performing image conversions, validating indices and saving images.
"""
from os import PathLike
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from . import DefaultBase, GenerationArgs


class ResultBase(DefaultBase):
    """ResultBase class is designed to handle and process the images"""

    image: Optional[
        Union[torch.Tensor, np.ndarray, Image.Image, List[Union[torch.Tensor, np.ndarray, Image.Image]]]
    ] = None
    output_type: Optional[Literal["np", "pt", "pil"]] = "pt"
    args: GenerationArgs
    samples_dir: Optional[PathLike] = "samples"
