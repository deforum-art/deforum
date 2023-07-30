import datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Literal, Tuple, Union

import cv2
import torch
from loguru import logger
from PIL import Image
from torchvision.io import ImageReadMode as _ImageReadMode
from torchvision.io import (
    decode_jpeg,
    decode_png,
    encode_jpeg,
    encode_png,
    read_file,
    read_image,
    write_jpeg,
    write_png,
)

from deforum.typed_classes import ResultBase
from deforum.utils.string_parsing import (
    TemplateParser,
    buffer_index_to_digits,
    find_next_index_in_template,
    normalize_text,
)


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


class ImageHandler:
    @classmethod
    def encode_image_as(cls, image, mode=ImageReadMode.RGB, jpeg_quality=95, png_compression=6, format="jpg"):
        if isinstance(image, torch.Tensor):
            if format == "jpg":
                return encode_jpeg(image, quality=jpeg_quality)
            elif format == "png":
                return encode_png(image, compression_level=png_compression)
            else:
                raise ValueError(f"Unknown image format {format}")
        else:
            raise ValueError(f"Unknown image type {type(image)}")

    @classmethod
    def to_bytes_torch(cls, image, mode=ImageReadMode.RGB, jpeg_quality=95, png_compression=6, format="jpg"):
        if isinstance(image, torch.Tensor):
            if format == "jpg":
                return encode_jpeg(image, quality=jpeg_quality).detach().cpu().numpy().tobytes()
            elif format == "png":
                return encode_png(image, compression_level=png_compression).detach().cpu().numpy().tobytes()
            else:
                raise ValueError(f"Unknown image format {format}")
        else:
            raise ValueError(f"Unknown image type {type(image)}")

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
    def write_image_torch(cls, image: torch.Tensor, path: PathLike, quality=95, compression_level=6):
        path = Path(path) if not isinstance(path, Path) else path
        if image.shape[-1] == 3:
            image = image.clone().permute(2, 0, 1)

        image = image.float()
        image = image - image.min()
        image = image / image.max()
        image = image * 255.0
        image = image.clamp(0, 255).detach().type(torch.uint8).cpu()
        if path.suffix == ".jpg" or path.suffix == ".jpeg":
            write_jpeg(image, path.as_posix(), quality=quality)
        elif path.suffix == ".png":
            write_png(image, path.as_posix(), compression_level=compression_level)
        else:
            raise ValueError(f"Invalid file extension {path.suffix}")

    @classmethod
    def check_output_type(cls, result: ResultBase):
        """
        Helper method to check the output type.
        Currently, only 'pt' (Pytorch) is supported
        """
        assert result.output_type == "pt", f"Only pytorch output type is supported for now, got {result.output_type}"

    @classmethod
    def check_index(cls, result: ResultBase, index: int):
        """Helper method to validate the index is within the bounds of image list"""
        assert index < len(result.image), f"Index {index} is out of bounds for image list of length {len(result.image)}"

    @classmethod
    def save_images(
        cls,
        result: ResultBase,
        template_str: Union[TemplateParser, str] = "samples/$custom_$timestr_$prompt_$index",
        image_index: int = -1,
        custom: str = "sample",
        quality: int = 95,
        template_index_key="index",
        index_digits=6,
        suffix=".jpg",
    ):
        """
        Save images in a custom directory and filename format.
        You can use `$custom`, `$timestr`, `$prompt`,and `$index` in the format string.
        To represent a directory, use slashes `/` in the format string.
        """

        if template_index_key.startswith("$"):
            template_index_key = template_index_key[1:]

        if not isinstance(template_str, TemplateParser):
            template_str = TemplateParser(template_str + suffix if not template_str.endswith(suffix) else template_str)
        if not template_str.template.endswith(suffix):
            template_str = TemplateParser(template_str.template + suffix)

        # Make sure index key is in template string
        assert "$" + template_index_key in template_str.template, (
            f"Template string must contain the template_index_key! key={template_index_key}, format_str={template_str.template}"
            + "\n(HINT: This is a template string, meaning you use '$' in a normal string to signify the substitution location of a template key location)"
        )

        # Generate timestamp
        time_date = datetime.datetime.fromtimestamp(result.args.start_time)
        timestr = time_date.strftime("%Y-%m-%dT%H-%M-%S")

        # Normalize prompt text
        promptstr = normalize_text(result.args.prompt if isinstance(result.args.prompt, str) else result.args.prompt[0])

        # Prepare the images to be saved
        assert (
            len(result.image) > image_index
        ), f"Image index {image_index} is out of bounds for image list of length {len(result.image)}"
        list_images = result.image if image_index < 0 else [result.image[image_index]]

        kwargs = dict(custom=custom, timestr=timestr, prompt=promptstr)
        kwargs[template_index_key] = 0

        idx = find_next_index_in_template(
            template_index_key=template_index_key,
            template_string=template_str,
            kwargs=kwargs,
            minimum_index=0,
        )
        # Loop over images and save each
        for i, image in enumerate(list_images):
            index = idx + i
            # Form the path (directory and filename)
            index = buffer_index_to_digits(index, index_digits)
            kwargs[template_index_key] = index
            file_path = template_str.safe_substitute(kwargs)
            logger.info(f"Saving image {i} to file_path={file_path}")
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            # Save the image
            ImageHandler.write_image_torch(image, file_path, quality=quality)

    @classmethod
    def to_bytes(cls, result: ResultBase, index: int = -1, quality: int = 95) -> bytes:
        """
        Method for converting image to bytes.
        """
        cls.check_output_type(result)
        cls.check_index(result, index)

        return ImageHandler.to_bytes_torch(result.image[index], jpeg_quality=quality)


def resize_tensor_result(
    result: ResultBase, new_size: Tuple[int, int], tensor_format: Literal["nchw", "hwc", "nhwc", "chw"] = "nhwc"
):
    """
    Resize the given image to the given size.

        args (GenerationArgs): The arguments used to generate the image.
        result (ResultBase): The image to resize.
        new_size (Tuple[int,int]): The new size of the image in (height, width) format.
        tensor_format (str): The format of the tensor. One of "nchw", "hwc", "nhwc", "chw".

    Returns:
        ResultBase: The resized image contained in a ResultBase object, with the new size in the args, and the resized image in the image field.
        Also, the image is in the "nchw" format.

    """
    if tensor_format == "nhwc":
        result.image = result.image.permute(0, 3, 1, 2)
    elif tensor_format == "hwc":
        result.image = result.image.permute(2, 0, 1).unsqueeze(0)
    elif tensor_format == "chw":
        result.image = result.image.unsqueeze(0)

    result.image = torch.nn.functional.interpolate(result.image, size=new_size, mode="bicubic", antialias=True)
    result.args.height = new_size[0]
    result.args.width = new_size[1]
    result.args.image = result.image
    return result
