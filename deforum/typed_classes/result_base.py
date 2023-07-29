from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from . import DefaultBase, GenerationArgs
from deforum.utils import normalize_text, ImageHandler


class ResultBase(DefaultBase):
    image: Optional[
        Union[torch.Tensor, np.ndarray, Image.Image, List[Union[torch.Tensor, np.ndarray, Image.Image]]]
    ] = None
    output_type: Optional[Literal["np", "pt", "pil"]] = "pt"
    args: GenerationArgs
    samples_dir: Optional[PathLike] = "samples"

    def save_images(
        self, samples_dir: PathLike = None, index: int = -1, format_str: str = "{}_{:06d}.jpg", quality: int = 95
    ) -> None:
        if samples_dir is None:
            samples_dir = self.samples_dir
        samples_dir = Path(samples_dir) if not isinstance(samples_dir, Path) else samples_dir
        samples_dir.mkdir(exist_ok=True, parents=True)
        # TODO: add support for multiple image types (just going to use tensor for now)
        timestr = self.args.start_time.strftime("%Y-%m-%dT%H-%M-%S")
        assert self.output_type == "pt", f"Only pt output type is supported for now, got {self.output_type}"
        if not (samples_dir / timestr).exists():
            (samples_dir / timestr).mkdir(exist_ok=True, parents=True)
        promptstr = normalize_text(self.args.prompt if isinstance(self.args.prompt, str) else self.args.prompt[0])
        if index < 0:
            for i, image in enumerate(self.image):
                ImageHandler.write_image_torch(
                    image, ((samples_dir / timestr) / format_str.format(promptstr, i)), quality=quality
                )
        else:
            assert index < len(self.image), f"Index {index} is out of bounds for image list of length {len(self.image)}"
            ImageHandler.write_image_torch(
                self.image[index], ((samples_dir / timestr) / format_str.format(promptstr, index)), quality=quality
            )

    def to_bytes(self, index: int = -1, quality: int = 95) -> bytes:
        assert self.output_type == "pt", f"Only pt output type is supported for now, got {self.output_type}"
        if index < 0:
            return ImageHandler.to_bytes_torch(self.image[0], quality=quality)
        else:
            assert index < len(self.image), f"Index {index} is out of bounds for image list of length {len(self.image)}"
            return ImageHandler.to_bytes_torch(self.image[index], quality=quality)
