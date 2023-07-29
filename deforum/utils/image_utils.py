from enum import Enum
from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.io import ImageReadMode as _ImageReadMode
from torchvision.io import decode_jpeg, decode_png, read_file, read_image, write_png, write_jpeg


class ImageReadMode(Enum):
    UNCHANGED = "UNCHANGED"
    GRAY = "GRAY"
    GRAY_ALPHA = "GRAY_ALPHA"
    RGB = "RGB"
    RGBA = "RGBA"
    BGR = "BGR"
    BGRA = "BGRA"
    L = "L"
    LA = "LA"
    P = "P"

    def channels(self):
        if self.name in ["RGB", "BGR"]:
            return 3
        elif self.name == "GRAY_ALPHA":
            return 2
        elif self.name.endswith("A"):
            return 4
        elif self.name == "GRAY":
            return 1
        else:
            return -1  # unknown aka UNCHANGED


class ImageReader:
    @classmethod
    def read_image_torch(cls, image, mode=ImageReadMode.RGB, device="cuda", return_hwc=False, to_numpy=False):
        image = image.as_posix() if isinstance(image, Path) else image
        mode = cls._map_read_mode_to_torch(mode)
        if image.lower().endswith(".jpg") or image.lower().endswith(".jpeg"):
            im = decode_jpeg(read_file(image), mode=mode, device=device)
        elif image.lower().endswith(".png"):
            im = decode_png(read_file(image), mode=mode).to(device)
        else:
            im = read_image(image, mode=mode).to(device)
        if return_hwc:
            if im.ndim == 3:
                im = im.permute(1, 2, 0)
        if to_numpy:
            return im.cpu().numpy()
        return im

    @classmethod
    def read_image_pil(cls, image, mode=ImageReadMode.RGB):
        image = image.as_posix() if isinstance(image, Path) else image
        im = Image.open(image).convert(mode.name if isinstance(mode, ImageReadMode) else mode)
        return im

    @classmethod
    def read_image_cv2(cls, image, mode=cv2.IMREAD_UNCHANGED):
        image = image.as_posix() if isinstance(image, Path) else image
        im = cv2.imread(image, mode)
        return im

    @classmethod
    def _map_read_mode_to_torch(cls, mode: ImageReadMode):
        if mode == ImageReadMode.GRAY:
            return _ImageReadMode.GRAY
        elif mode == ImageReadMode.GRAY_ALPHA:
            return _ImageReadMode.GRAY_ALPHA
        elif mode == ImageReadMode.RGB:
            return _ImageReadMode.RGB
        elif mode == ImageReadMode.RGBA:
            return _ImageReadMode.RGB_ALPHA
        elif mode == ImageReadMode.BGR:
            return _ImageReadMode.RGB
        elif mode == ImageReadMode.BGRA:
            return _ImageReadMode.RGB_ALPHA
        else:
            raise ValueError(f"Invalid mode {mode}")

    @classmethod
    def write_image_torch(cls, image: torch.Tensor, path: PathLike):
        path = Path(path) if not isinstance(path, Path) else path
        if image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        image = image.float()
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        image = image.clamp(0, 255).detach().type(torch.uint8).cpu()
        if path.suffix == ".jpg" or path.suffix == ".jpeg":
            write_jpeg(image, path.as_posix())
        elif path.suffix == ".png":
            write_png(image, path.as_posix())
        else:
            raise ValueError(f"Invalid file extension {path.suffix}")
