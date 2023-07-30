"""
This module defines a class ResultBase that handles operations related to image objects.
Operations include checking output types, performing image conversions, validating indices and saving images.
"""
import datetime
from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from PIL import Image
import torch

from deforum.utils import normalize_text, ImageHandler
from . import DefaultBase, GenerationArgs

MAX_ATTEMPTS = 1000000


class ResultBase(DefaultBase):
    """ResultBase class is designed to handle and process the images"""

    image: Optional[
        Union[torch.Tensor, np.ndarray, Image.Image, List[Union[torch.Tensor, np.ndarray, Image.Image]]]
    ] = None
    output_type: Optional[Literal["np", "pt", "pil"]] = "pt"
    args: GenerationArgs
    samples_dir: Optional[PathLike] = "samples"

    def check_output_type(self):
        """
        Helper method to check the output type.
        Currently, only 'pt' (Pytorch) is supported
        """
        assert self.output_type == "pt", f"Only pytorch output type is supported for now, got {self.output_type}"

    def check_index(self, index: int):
        """Helper method to validate the index is within the bounds of image list"""
        assert index < len(self.image), f"Index {index} is out of bounds for image list of length {len(self.image)}"

    def save_images(
        self,
        suffix: str = ".jpg",
        pattern: str = "samples/{custom}_{index}",
        custom: str = "sample",
        quality: int = 95,
        compression_level: int = 6,
    ) -> List[str]:
        """
        Save images in a custom directory and filename format.
        You can use `{custom}`, `{timestr}`, `{prompt}`, and `{index}` in the filename pattern.
        To represent a directory, use slashes `/` in the filename pattern.
        """
        time_date = datetime.datetime.fromtimestamp(self.args.start_time)
        timestr = time_date.strftime("%Y-%m-%dT%H-%M-%S")
        promptstr = normalize_text(self.args.prompt if isinstance(self.args.prompt, str) else self.args.prompt[0])
        pattern = pattern.replace("{index}", "{index:06d}")
        Path(pattern.format(custom=custom, timestr=timestr, prompt=promptstr, index=0)).parent.mkdir(
            parents=True, exist_ok=True
        )  # Creating all the parent directories in the pattern

        file_paths = []
        for i, image in enumerate(self.image):
            for index in range(MAX_ATTEMPTS):  # hardcoding max attempts to 10000 to avoid potential infinite loop
                file_path_str = pattern.format(custom=custom, timestr=timestr, prompt=promptstr, index=index)
                file_path = Path(file_path_str).with_suffix(suffix)
                if not file_path.exists():
                    break
                elif "{index:06d}" not in pattern:
                    pattern = pattern + "_{index:06d}"

            # raise an error if while loop ended normally (no break statement encountered)
            else:
                raise RuntimeError(f"Exceeded max attempts ({MAX_ATTEMPTS}) to save file at path: {file_path}")

            ImageHandler.write_image_torch(image, file_path, quality=quality, compression_level=compression_level)
            file_paths.append(str(file_path))

        return file_paths

    def to_bytes(self, index: int = -1, quality: int = 95) -> bytes:
        """
        Method for converting image to bytes.
        """
        self.check_output_type()
        self.check_index(index)

        return ImageHandler.to_bytes_torch(self.image[index], jpeg_quality=quality)
